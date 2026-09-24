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

"""Reusable fixtures and strict contract examples for cold Database Replica tests."""

from __future__ import annotations

import json
import os
import pwd
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.preprocessing_execution import LocalExecutionFixture, preprocessing_execution_fixture

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    DATABASE_SOURCE_ROOT,
    DatabaseAccessPolicy,
    DatabaseProfileStagingSnapshot,
    DatabaseSetSelection,
    build_preprocessing_database_binding,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabaseSourceMountFacts,
    DatabaseSourceObservation,
)
from bspp.orchestration.contract.database_replica import (
    DatabaseCacheMountFacts,
    DatabaseCapacityGate,
    DatabaseReplicaCopyEvidence,
    DatabaseReplicaManifest,
    DatabaseReplicaMember,
    DatabaseRsyncOutcome,
)
from bspp.orchestration.contract.database_set_provisioning import (
    DatabaseAlias,
    DatabaseSetIdentity,
    DatabaseSourceManifest,
    DatabaseSourceMember,
    canonical_database_source_manifest_bytes,
)
from bspp.orchestration.runtime.preprocessing._database_placement_paths import (
    PRODUCTION_DATABASE_PLACEMENT_PATHS,
    DatabasePlacementPaths,
)


@pytest.fixture
def cold_cache_root(request: pytest.FixtureRequest) -> Path:
    root = Path(tempfile.mkdtemp(prefix="bspp-issue91-", dir="/dev/shm"))
    request.addfinalizer(lambda: _remove_test_cache(root))
    return root


def _remove_test_cache(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        path.chmod(0o700)
    root.chmod(0o700)
    shutil.rmtree(root)


def _cache_mount() -> DatabaseCacheMountFacts:
    return DatabaseCacheMountFacts(
        mount_id=52,
        parent_mount_id=31,
        device_major=0,
        device_minor=72,
        mount_root="/",
        mount_point=DATABASE_CACHE_ROOT,
        filesystem_type="tmpfs",
        mount_source="shm",
        mount_options=("rw",),
        super_options=("rw",),
        read_write=True,
        writable_verified=True,
    )


def _capacity() -> DatabaseCapacityGate:
    return DatabaseCapacityGate(
        available_user_bytes=128,
        allocated_replica_bytes=64,
        reserved_bytes=64,
        required_bytes=128,
        decision="sufficient",
    )


def _source_members() -> tuple[DatabaseSourceMember, ...]:
    return tuple(
        DatabaseSourceMember(
            role=role,
            database_name=name,
            logical_name=name,
            source_path=name,
            source_kind="regular",
            resolved_path=name,
            resolved_kind="regular",
            size_bytes=32,
            mtime_ns=123,
            alias_topology=(),
            preexisting_checksum=None,
        )
        for role, name in (("primary", "primary"), ("metagenomic", "metagenomic"))
    )


def _source_observation() -> DatabaseSourceObservation:
    return DatabaseSourceObservation(
        source_container_root=DATABASE_SOURCE_ROOT,
        source_manifest_sha256="a" * 64,
        members=_source_members(),
    )


def _source_mount() -> DatabaseSourceMountFacts:
    return DatabaseSourceMountFacts(
        mount_id=42,
        parent_mount_id=31,
        device_major=0,
        device_minor=29,
        mount_root="/",
        mount_point=DATABASE_SOURCE_ROOT,
        filesystem_type="overlay",
        mount_source="overlay",
        mount_options=("ro",),
        super_options=("ro",),
        read_only=True,
    )


def _copy_evidence() -> DatabaseReplicaCopyEvidence:
    outcomes = tuple(
        DatabaseRsyncOutcome(
            resolved_source_path=name,
            destination_logical_name=name,
            logical_names=(name,),
            size_bytes=32,
            argv=(
                "/usr/bin/rsync",
                "--archive",
                "--no-owner",
                "--no-group",
                "--no-perms",
                "--protect-args",
                f"{DATABASE_SOURCE_ROOT}/{name}",
                f"{DATABASE_CACHE_ROOT}/replicas/.population-<private>/{name}",
            ),
            exit_code=0,
        )
        for name in ("metagenomic", "primary")
    )
    return DatabaseReplicaCopyEvidence(
        source_count=2,
        selected_workers=2,
        total_copied_bytes=64,
        elapsed_nanoseconds=16,
        aggregate_bytes_per_second=4_000_000_000,
        outcomes=outcomes,
    )


def _manifest() -> DatabaseReplicaManifest:
    member = DatabaseReplicaMember(
        role="primary",
        database_name="primary",
        logical_name="primary",
        resolved_source_path="primary",
        replica_path="primary",
        size_bytes=32,
        mtime_ns=123,
        hardlink_group="primary",
    )
    metagenomic = DatabaseReplicaMember(
        role="metagenomic",
        database_name="metagenomic",
        logical_name="metagenomic",
        resolved_source_path="metagenomic",
        replica_path="metagenomic",
        size_bytes=32,
        mtime_ns=123,
        hardlink_group="metagenomic",
    )
    return DatabaseReplicaManifest(
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        source_manifest_sha256="a" * 64,
        members=(metagenomic, member),
        pre_copy_source_observation=_source_observation(),
        post_copy_source_observation=_source_observation(),
        copy_evidence=_copy_evidence(),
    )


def _stage_required_fixture(
    root: Path,
    cache_profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    unix_user: str | None = None,
    requested_policy: DatabaseAccessPolicy = DatabaseAccessPolicy.STAGE_REQUIRED,
) -> tuple[LocalExecutionFixture, Path, Path]:
    fixture = preprocessing_execution_fixture(root, direct_policy=False)
    original = fixture.runspec.payload.database
    action = fixture.action.payload
    selected_unix_user = unix_user or pwd.getpwuid(os.geteuid()).pw_name
    staging = DatabaseProfileStagingSnapshot(
        cache_root=str(cache_profile_root),
        unix_user=selected_unix_user,
        expected_filesystem_type="tmpfs",
        reserve_bytes=0,
        lock_wait_seconds=60,
    )
    binding = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=original.database_set,
            requested_policy=requested_policy,
        ),
        source_manifest=original.source_manifest,
        source_manifest_projection=original.source_manifest_projection,
        staging=staging,
        gpuserver_argv=action.gpuserver_argv,
        search_argv=action.search_argv,
    )
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=binding))
    fixture.runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    fixture = replace(fixture, runspec=runspec)
    cache_root = Path(staging.user_cache_root)
    cache_root.mkdir(parents=True)
    manifest_path = root / "database-source-manifest.json"
    manifest_path.write_bytes(canonical_database_source_manifest_bytes(binding.source_manifest))
    mountinfo_path = root / "mountinfo"
    source_device = fixture.database_source_root.stat().st_dev
    cache_device = cache_root.stat().st_dev
    mountinfo_path.write_text(
        f"42 31 {os.major(source_device)}:{os.minor(source_device)} / {DATABASE_SOURCE_ROOT} ro - "
        "overlay overlay ro\n"
        f"52 31 {os.major(cache_device)}:{os.minor(cache_device)} / {DATABASE_CACHE_ROOT} rw - "
        "tmpfs shm rw\n"
    )
    fixture = replace(
        fixture,
        database_placement_paths=DatabasePlacementPaths(
            source_root=fixture.database_source_root,
            cache_root=cache_root,
            selected_root=fixture.database_source_root,
            lease=root / "database-replica-lease",
            mountinfo=mountinfo_path,
            rsync=PRODUCTION_DATABASE_PLACEMENT_PATHS.rsync,
        ),
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(fixture.action.resources.cpus_per_task))
    return fixture, manifest_path, cache_root


def _replace_stage_manifest(
    fixture: LocalExecutionFixture,
    manifest: DatabaseSourceManifest,
    manifest_path: Path,
) -> LocalExecutionFixture:
    original = fixture.runspec.payload.database
    action = fixture.action.payload
    binding = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=manifest.database_set,
            requested_policy=DatabaseAccessPolicy.STAGE_REQUIRED,
        ),
        source_manifest=manifest,
        source_manifest_projection=original.source_manifest_projection,
        staging=original.staging,
        gpuserver_argv=action.gpuserver_argv,
        search_argv=action.search_argv,
    )
    runspec = replace(fixture.runspec, payload=replace(fixture.runspec.payload, database=binding))
    fixture.runspec_path.chmod(0o600)
    fixture.runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n")
    manifest_path.write_bytes(canonical_database_source_manifest_bytes(manifest))
    return replace(fixture, runspec=runspec)


def _shared_source_fixture(
    root: Path,
    cache_profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LocalExecutionFixture, Path, Path]:
    fixture, manifest_path, cache_root = _stage_required_fixture(root, cache_profile_root, monkeypatch)
    original = fixture.runspec.payload.database.source_manifest
    shared_name = "shared-database-member"
    members = tuple(
        replace(
            member,
            source_kind="symlink",
            resolved_path=shared_name,
            alias_topology=(DatabaseAlias(path=member.source_path, target=shared_name),),
        )
        for member in original.members
    )
    manifest = replace(original, members=members)
    for member in original.members:
        (fixture.database_source_root / member.source_path).unlink()
    shared = fixture.database_source_root / shared_name
    shared.write_bytes(b"x")
    os.utime(shared, ns=(1, 1))
    shared.chmod(0o444)
    for member in members:
        (fixture.database_source_root / member.source_path).symlink_to(shared_name)
    fixture = _replace_stage_manifest(fixture, manifest, manifest_path)
    return fixture, manifest_path, cache_root
