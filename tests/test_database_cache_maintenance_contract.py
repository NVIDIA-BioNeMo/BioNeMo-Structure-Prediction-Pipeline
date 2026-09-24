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

"""Contract tests for the dedicated acceptance-cache maintenance projection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.database_cache_maintenance import (
    DatabaseAcceptanceCacheProfile,
    DatabaseCacheRemovedEntry,
    canonical_database_cache_maintenance_profile_bytes,
    removed_database_cache_entries_digest,
    select_database_acceptance_cache_profile,
)
from bspp.orchestration.contract.database_placement import (
    database_profile_staging_snapshot_from_mapping,
    database_set_selection_from_mapping,
)
from bspp.orchestration.control.profiles import resolve_cluster_profile


def test_shared_selector_projects_only_complete_acceptance_cache_authority(tmp_path: Path) -> None:
    root = tmp_path / "acceptance" / "cache" / "root"
    raw = {
        "clusters": {
            "example-cluster-acceptance": _profile(root),
        }
    }

    selected = select_database_acceptance_cache_profile(raw, "example-cluster-acceptance")

    assert selected == DatabaseAcceptanceCacheProfile(
        profile_name="example-cluster-acceptance",
        database_cache_namespace="acceptance",
        cache_root=str(root),
        unix_user="agent",
        expected_filesystem_type="overlay",
        lock_wait_seconds=3,
    )
    assert json.loads(canonical_database_cache_maintenance_profile_bytes(selected)) == selected.to_mapping()
    assert "owner" not in selected.to_mapping()
    assert "database_cache_reserve_bytes" not in selected.to_mapping()


@pytest.mark.parametrize("key", ["extends", "inherits", "parent", "base_profile"])
def test_shared_selector_centrally_rejects_profile_inheritance(tmp_path: Path, key: str) -> None:
    profile = _profile(tmp_path / "acceptance" / "cache" / "root")
    profile[key] = "base"

    with pytest.raises(ValueError, match="inheritance"):
        select_database_acceptance_cache_profile({"clusters": {"selected": profile}}, "selected")


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"database_cache_namespace": None}, "acceptance"),
        ({"database_cache_namespace": "production"}, "acceptance"),
        ({"database_cache_root": "/"}, "broad"),
        ({"database_cache_root": "/tmp/cache"}, "shallow"),
        ({"database_cache_root": "/tmp/../var/acceptance/cache"}, "normalize"),
        ({"database_cache_root": "relative/cache/root"}, "absolute"),
        ({"database_cache_unix_user": "../other"}, "Unix user"),
        ({"database_cache_filesystem_type": ""}, "filesystem"),
        ({"database_lock_wait_seconds": 0}, "positive"),
        ({"database_cache_reserve_bytes": None}, "complete"),
    ],
)
def test_shared_selector_rejects_incomplete_or_broad_authority(
    tmp_path: Path,
    update: dict[str, object],
    message: str,
) -> None:
    profile = _profile(tmp_path / "acceptance" / "cache" / "root")
    profile.update(update)

    with pytest.raises((TypeError, ValueError), match=message):
        select_database_acceptance_cache_profile({"clusters": {"selected": profile}}, "selected")


def test_shared_selector_rejects_alias_to_any_sibling_profile(tmp_path: Path) -> None:
    root = tmp_path / "acceptance" / "cache" / "root"
    selected = _profile(root)
    production = _profile(root)
    production["database_cache_namespace"] = None

    with pytest.raises(ValueError, match="aliases"):
        select_database_acceptance_cache_profile(
            {"clusters": {"acceptance": selected, "production": production}},
            "acceptance",
        )


def test_removed_entry_digest_binds_complete_sorted_filesystem_facts() -> None:
    entries = (
        DatabaseCacheRemovedEntry(
            kind="replica",
            source_manifest_sha256="b" * 64,
            basename="b" * 64,
            device=17,
            inode=29,
        ),
        DatabaseCacheRemovedEntry(
            kind="population",
            source_manifest_sha256="a" * 64,
            basename=f".population-{'a' * 64}-{'c' * 32}",
            device=17,
            inode=23,
        ),
    )

    assert removed_database_cache_entries_digest(entries) == (
        "d9a6da36006bcc9c62d0dab30d89a1afe3e64464156fe36d0c4b36dddefdb3fe"
    )
    changed_inode = (
        entries[0],
        DatabaseCacheRemovedEntry(
            kind="population",
            source_manifest_sha256="a" * 64,
            basename=f".population-{'a' * 64}-{'c' * 32}",
            device=17,
            inode=31,
        ),
    )
    assert removed_database_cache_entries_digest(changed_inode) != (
        "d9a6da36006bcc9c62d0dab30d89a1afe3e64464156fe36d0c4b36dddefdb3fe"
    )


def test_control_exposes_the_exact_shared_projection(tmp_path: Path) -> None:
    root = tmp_path / "acceptance" / "cache" / "root"
    user = _profile(root)
    user.update(
        {
            "transport": "local-slurm",
            "database_access_policies": ["stage-required"],
            "project_root": "/project",
            "output_root": "/output",
            "staging_root": "/staging",
            "afdb_toolkit_repo": "/toolkit",
            "orchestration_repo": "/orchestration",
            "image": "/image.sqsh",
        }
    )
    config = {"clusters": {"example-cluster": user}}
    config_path = _write_yaml(tmp_path / "profiles.yaml", config)
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "account": "acct",
                    "resources": {
                        "gpu_worker": {
                            "partition": "gpu",
                            "cpus_per_task": 1,
                            "memory": "1G",
                            "time": "00:01:00",
                            "gres": "gpu:1",
                        }
                    },
                }
            }
        },
    )

    resolved = resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    assert resolved.database_acceptance_cache == select_database_acceptance_cache_profile(config, "example-cluster")


@pytest.mark.parametrize(
    "forbidden",
    [
        {"database_cache_namespace": "acceptance"},
        {"maintenance_requested": True},
        {"database_cache_root": "/operator/override/cache"},
    ],
)
def test_phase_plan_database_selection_rejects_maintenance_authority(forbidden: dict[str, object]) -> None:
    selection: dict[str, object] = {
        "database_set": {"identifier": "bspp-search", "version": "2026-08"},
        **forbidden,
    }

    with pytest.raises(ValueError, match="unknown"):
        database_set_selection_from_mapping(selection)


@pytest.mark.parametrize("forbidden_key", ["database_cache_namespace", "maintenance_requested"])
def test_phase_runspec_staging_snapshot_rejects_maintenance_authority(forbidden_key: str) -> None:
    staging: dict[str, object] = {
        "cache_root": "/node/cache/bspp",
        "expected_filesystem_type": "xfs",
        "lock_wait_seconds": 30,
        "reserve_bytes": 0,
        "unix_user": "agent",
        forbidden_key: "acceptance" if forbidden_key == "database_cache_namespace" else True,
    }

    with pytest.raises(ValueError, match="unknown"):
        database_profile_staging_snapshot_from_mapping(staging)


def test_runtime_does_not_depend_on_control_and_image_bakes_control_wheel() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_project = (root / "packages/orchestration-runtime/pyproject.toml").read_text()
    entrypoint = (root / "containers/scripts/entrypoint.sh").read_text()
    dockerfile = (root / "containers/preprocessing/Dockerfile").read_text()
    runtime_sources = "\n".join(
        path.read_text() for path in sorted((root / "packages/orchestration-runtime/src").rglob("*.py"))
    )

    assert '"bspp-orchestration-control"' not in runtime_project
    assert "packages/orchestration-control" not in entrypoint
    assert "bspp.orchestration.control" not in runtime_sources
    assert 'importlib.metadata.version("bspp-orchestration-control")' in dockerfile


def _profile(root: Path) -> dict[str, object]:
    return {
        "owner": "ignored-by-maintenance-projection",
        "database_cache_namespace": "acceptance",
        "database_cache_root": str(root),
        "database_cache_unix_user": "agent",
        "database_cache_filesystem_type": "overlay",
        "database_cache_reserve_bytes": 0,
        "database_lock_wait_seconds": 3,
    }


def _write_yaml(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=True))
    return path
