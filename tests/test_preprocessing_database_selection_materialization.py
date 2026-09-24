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

"""Public contract tests for preprocessing database selection materialization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
    DatabaseAccessPolicy,
    DatabaseSetSelection,
    build_preprocessing_database_binding,
    database_set_selection_from_mapping,
    preprocessing_database_binding_from_mapping,
)
from tests.support.preprocessing_execution import preprocessing_execution_fixture


def test_database_set_selection_defaults_to_stage_required() -> None:
    selection = database_set_selection_from_mapping(
        {"database_set": {"identifier": "bspp-search", "version": "2026-08"}}
    )

    assert selection.database_set.identifier == "bspp-search"
    assert selection.database_set.version == "2026-08"
    assert selection.requested_policy == DatabaseAccessPolicy.STAGE_REQUIRED


def test_database_set_selection_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Database Set selection fields"):
        database_set_selection_from_mapping(
            {
                "database_set": {"identifier": "bspp-search", "version": "2026-08"},
                "database_root": "/databases",
            }
        )


@pytest.mark.parametrize(
    ("policy", "expected_branches"),
    [
        (DatabaseAccessPolicy.STAGE_REQUIRED, ("staged",)),
        (
            DatabaseAccessPolicy.STAGE_PREFERRED,
            ("staged", "direct-capacity-fallback"),
        ),
        (DatabaseAccessPolicy.DIRECT, ("direct-requested",)),
    ],
)
def test_database_binding_materializes_exact_policy_branch_closure(
    tmp_path: Path,
    policy: DatabaseAccessPolicy,
    expected_branches: tuple[str, ...],
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    source = fixture.runspec.payload.database
    action = fixture.action.payload
    binding = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=source.database_set,
            requested_policy=policy,
        ),
        source_manifest=source.source_manifest,
        source_manifest_projection="attempts/attempt-0001/database-source-manifest.json",
        staging=source.staging if policy != DatabaseAccessPolicy.DIRECT else None,
        gpuserver_argv=action.gpuserver_argv,
        search_argv=action.search_argv,
    )

    assert tuple(branch.branch_kind for branch in binding.branches) == expected_branches
    assert binding.selected_container_root == SELECTED_DATABASE_ROOT
    assert all(not branch.finalization_mounts for branch in binding.branches)
    assert all(branch.gpuserver_argv == action.gpuserver_argv for branch in binding.branches)
    assert all(branch.search_argv == action.search_argv for branch in binding.branches)
    for branch in binding.branches:
        assert branch.placement_mounts[0].target == DATABASE_SOURCE_ROOT
        if branch.branch_kind == "staged":
            assert branch.scientific_mounts[0].target == SELECTED_DATABASE_ROOT
            assert branch.placement_mounts[1].target == DATABASE_CACHE_ROOT
            assert branch.placement_mounts[1].read_only is False
            assert branch.scientific_mounts[0].purpose == "selected-replica"
            assert branch.scientific_mounts[1].purpose == "replica-lease"
            assert branch.scientific_mounts[1].source == (
                f"{source.staging.user_cache_root}/.locks/{binding.source_manifest_sha256}.lock"
            )
            assert branch.scientific_mounts[1].target == "/run/bspp/database/replica-lease.lock"
            assert branch.scientific_mounts[1].read_only is True
        else:
            assert len(branch.placement_mounts) == 1
            assert branch.scientific_mounts
            assert all(mount.purpose == "selected-source" for mount in branch.scientific_mounts)
            assert all(mount.source_kind == "file" for mount in branch.scientific_mounts)
            assert all(Path(mount.target).parent == Path(SELECTED_DATABASE_ROOT) for mount in branch.scientific_mounts)
            assert {Path(mount.target).name for mount in branch.scientific_mounts} == {
                member.logical_name for member in source.source_manifest.members
            }

    assert preprocessing_database_binding_from_mapping(binding.to_mapping()) == binding


def test_database_binding_rejects_missing_staging_and_tampered_commands(tmp_path: Path) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    source = fixture.runspec.payload.database
    action = fixture.action.payload
    with pytest.raises(ValueError, match="staged database branch requires"):
        build_preprocessing_database_binding(
            selection=DatabaseSetSelection(database_set=source.database_set),
            source_manifest=source.source_manifest,
            source_manifest_projection="attempts/attempt-0001/database-source-manifest.json",
            staging=None,
            gpuserver_argv=action.gpuserver_argv,
            search_argv=action.search_argv,
        )

    preferred = build_preprocessing_database_binding(
        selection=DatabaseSetSelection(
            database_set=source.database_set,
            requested_policy=DatabaseAccessPolicy.STAGE_PREFERRED,
        ),
        source_manifest=source.source_manifest,
        source_manifest_projection="attempts/attempt-0001/database-source-manifest.json",
        staging=source.staging,
        gpuserver_argv=action.gpuserver_argv,
        search_argv=action.search_argv,
    )
    mapping = deepcopy(preferred.to_mapping())
    mapping["branches"][1]["search_argv"][-1] = "tampered"
    with pytest.raises(ValueError, match="identical Scientific Kernel commands"):
        preprocessing_database_binding_from_mapping(mapping)


def test_staged_database_binding_isolated_to_frozen_unix_user_namespace(tmp_path: Path) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    binding = fixture.runspec.payload.database
    staging = binding.staging
    assert staging is not None

    assert staging.unix_user == "tester"
    staged = binding.branches[0]
    cache_mount = next(item for item in staged.placement_mounts if item.purpose == "cache")
    replica_mount = next(item for item in staged.scientific_mounts if item.purpose == "selected-replica")
    assert cache_mount.source == "/var/cache/bspp/local-test/users/tester"
    assert cache_mount.read_only is False
    assert replica_mount.source == (
        f"/var/cache/bspp/local-test/users/tester/replicas/{binding.source_manifest_sha256}"
    )
    assert all(item.source != staging.cache_root for item in staged.placement_mounts)
    assert staging.cache_root in binding.identity_protected_targets
    writable_sources = [
        mount.source
        for branch in binding.branches
        for mounts in (branch.placement_mounts, branch.scientific_mounts, branch.finalization_mounts)
        for mount in mounts
        if not mount.read_only
    ]
    assert writable_sources == [staging.user_cache_root]


@pytest.mark.parametrize("unix_user", ["../other", "alice/bob", "", "."])
def test_database_binding_loader_rejects_unsafe_unix_user_identity(
    tmp_path: Path,
    unix_user: str,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    mapping = deepcopy(fixture.runspec.payload.database.to_mapping())
    staging_mapping = mapping["staging"]
    assert isinstance(staging_mapping, dict)
    staging_mapping["unix_user"] = unix_user

    with pytest.raises(ValueError, match="unix_user"):
        preprocessing_database_binding_from_mapping(mapping)


def test_database_binding_loader_rejects_safe_unix_user_tamper_without_rematerialized_mounts(
    tmp_path: Path,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    mapping = deepcopy(fixture.runspec.payload.database.to_mapping())
    staging_mapping = mapping["staging"]
    assert isinstance(staging_mapping, dict)
    staging_mapping["unix_user"] = "other-user"

    with pytest.raises(ValueError, match="mounts do not match protected placement authority"):
        preprocessing_database_binding_from_mapping(mapping)


@pytest.mark.parametrize("mutation", ["swapped", "duplicate", "forged-target", "forged-mode"])
def test_staged_database_binding_rejects_swapped_duplicate_or_forged_lease_descriptor(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = preprocessing_execution_fixture(tmp_path / "work", direct_policy=False)
    mapping = deepcopy(fixture.runspec.payload.database.to_mapping())
    branch = mapping["branches"][0]
    assert isinstance(branch, dict)
    mounts = branch["scientific_mounts"]
    assert isinstance(mounts, list)
    if mutation == "swapped":
        mounts.reverse()
    elif mutation == "duplicate":
        mounts[1] = deepcopy(mounts[0])
    else:
        lease = mounts[1]
        assert isinstance(lease, dict)
        if mutation == "forged-target":
            lease["target"] = "/run/bspp/database/forged.lock"
        else:
            lease["read_only"] = False

    with pytest.raises(ValueError, match=r"protected placement authority|read-only"):
        preprocessing_database_binding_from_mapping(mapping)
