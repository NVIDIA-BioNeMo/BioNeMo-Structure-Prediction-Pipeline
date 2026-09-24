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

"""Tests for Cluster Profile resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.database_placement import (
    DatabaseAccessPolicy,
    DatabaseSetSelection,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.control.profiles import (
    resolve_cluster_profile,
    resolve_database_manifest_path,
)


def test_resolve_cluster_profile_merges_template_defaults_with_user_overrides(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "account": "template-account",
                    "resources": {
                        "gpu_worker": {
                            "partition": "gpu",
                            "cpus_per_task": 8,
                            "memory": "64G",
                            "time": "01:00:00",
                            "gres": "gpu:1",
                        },
                        "analysis_finalize": {
                            "partition": "cpu",
                            "cpus_per_task": 4,
                            "memory": "16G",
                            "time": "00:30:00",
                        },
                    },
                }
            }
        },
    )
    config_path = _write_yaml(
        tmp_path / "profiles.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "owner": "tester",
                    "database_access_policies": ["direct"],
                    "transport": "ssh",
                    "ssh_target": "example-cluster-login",
                    "runtime_qualification_expires_hours": 168,
                    "account": "user-account",
                    "paths": {
                        "project_root": "/project",
                        "output_root": "/output",
                        "staging_root": "/staging",
                        "afdb_toolkit_repo": "/toolkit",
                        "orchestration_repo": "/orchestration",
                        "image": "/images/bspp.sqsh",
                        "probe_root": "/tmp/probes",
                        "source_bundle_root": "/bundles",
                        "runtime_image_cache_root": "/image-cache",
                        "runtime_qualification_root": "/qualifications",
                        "runtime_qualification_control_root": "/workstation/qualifications",
                    },
                    "resources": {"gpu_worker": {"cpus_per_task": 30}},
                    "extra_mounts": [{"source": "/scratch", "target": "/scratch"}],
                }
            }
        },
    )

    profile = resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    assert profile.name == "example-cluster"
    assert profile.owner == "tester"
    assert profile.account == "user-account"
    assert profile.project_root == "/project"
    assert profile.image == "/images/bspp.sqsh"
    assert profile.runtime_image_policy == "digest-checked"
    assert profile.transport == "ssh"
    assert profile.ssh_target == "example-cluster-login"
    assert profile.probe_root == "/tmp/probes"
    assert profile.source_bundle_root == "/bundles"
    assert profile.runtime_image_cache_root == "/image-cache"
    assert profile.runtime_qualification_root == "/qualifications"
    assert profile.runtime_qualification_control_root == "/workstation/qualifications"
    assert profile.governed_package_root == "/qualifications/packages"
    assert profile.runtime_qualification_expires_hours == 168
    assert profile.extra_mounts[0].target == "/scratch"
    assert profile.resources["gpu_worker"].partition == "gpu"
    assert profile.resources["gpu_worker"].cpus_per_task == 30
    assert profile.resources["analysis_finalize"].memory == "16G"


def test_resolve_cluster_profile_rejects_conflicting_flat_and_nested_paths(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    config_path = _write_yaml(
        tmp_path / "profiles.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "owner": "tester",
                    "project_root": "/flat-project",
                    "transport": "local-slurm",
                    "paths": {
                        "project_root": "/nested-project",
                        "output_root": "/output",
                        "staging_root": "/staging",
                        "afdb_toolkit_repo": "/toolkit",
                        "orchestration_repo": "/orchestration",
                        "image": "/images/bspp.sqsh",
                    },
                }
            }
        },
    )

    with pytest.raises(ValueError, match="cannot be set both flat and under 'paths'"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_defaults_runtime_qualification_expiration(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    config_path = _write_yaml(
        tmp_path / "profiles.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "owner": "tester",
                    "database_access_policies": ["direct"],
                    "transport": "local-slurm",
                    "paths": {
                        "project_root": "/project",
                        "output_root": "/output",
                        "staging_root": "/staging",
                        "afdb_toolkit_repo": "/toolkit",
                        "orchestration_repo": "/orchestration",
                        "image": "/images/bspp.sqsh",
                        "runtime_qualification_root": "/qualifications",
                    },
                }
            }
        },
    )

    profile = resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    assert profile.runtime_qualification_expires_hours == 168
    assert profile.runtime_qualification_control_root == "/qualifications"


def test_resolve_cluster_profile_requires_absolute_control_root_for_ssh(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    base = {
        "owner": "tester",
        "database_access_policies": ["direct"],
        "transport": "ssh",
        "ssh_target": "login",
        "paths": {
            "project_root": "/project",
            "output_root": "/output",
            "staging_root": "/staging",
            "afdb_toolkit_repo": "/toolkit",
            "orchestration_repo": "/orchestration",
            "image": "/image.sqsh",
            "runtime_qualification_root": "/remote/qualifications",
        },
    }
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": base}})
    with pytest.raises(ValueError, match="requires runtime_qualification_control_root"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    base["paths"]["runtime_qualification_control_root"] = "relative/control"
    config_path = _write_yaml(tmp_path / "profiles-relative.yaml", {"clusters": {"example-cluster": base}})
    with pytest.raises(ValueError, match="must be absolute"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_ssh_runtime_qualification_requires_a_distinct_workstation_control_root(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    profile = _profile_data()
    profile.update(
        {
            "transport": "ssh",
            "ssh_target": "example-cluster-login",
            "runtime_qualification_root": "/cluster/qualifications",
        }
    )
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": profile}})

    with pytest.raises(ValueError, match="SSH Cluster Profile requires runtime_qualification_control_root"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_uses_bundled_template_when_template_path_is_absent(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "profiles.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "owner": "tester",
                    "database_access_policies": ["direct"],
                    "project_root": "/project",
                    "output_root": "/output",
                    "staging_root": "/staging",
                    "afdb_toolkit_repo": "/toolkit",
                    "orchestration_repo": "/orchestration",
                    "image": "/images/bspp.sqsh",
                    "transport": "local-slurm",
                }
            }
        },
    )

    profile = resolve_cluster_profile("example-cluster", config_path=config_path)

    assert profile.account == "example-account"
    assert profile.resources["gpu_worker"].partition == "example-gpu"


def test_name_keyed_explicit_template_wins_over_generic_shape(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {
            "clusters": {
                "example": {"account": "example-account", "resources": {}},
                "example-cluster": {
                    "account": "explicit-example-cluster-account",
                    "resources": {
                        "gpu_worker": {
                            "partition": "explicit-gpu",
                            "cpus_per_task": 8,
                            "memory": "32G",
                            "time": "01:00:00",
                            "gres": "gpu:1",
                        }
                    },
                },
            }
        },
    )
    data = _profile_data()
    data["transport"] = "local-slurm"
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    profile = resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    assert profile.account == "explicit-example-cluster-account"
    assert profile.resources["gpu_worker"].partition == "explicit-gpu"


def test_bundled_example_cluster_acceptance_profile_keeps_scheduler_defaults_separate(tmp_path: Path) -> None:
    ordinary = _profile_data()
    ordinary.update({"transport": "local-slurm"})
    acceptance = _staging_database_profile_data(tmp_path)
    acceptance.update(
        {
            "database_cache_namespace": "acceptance",
            "database_cache_root": "/node-cache/users/tester/bspp-acceptance",
            "resources": {"gpu_worker": {"nodelist": "gpu-node-017"}},
        }
    )
    config_path = _write_yaml(
        tmp_path / "profiles.yaml",
        {"clusters": {"example-cluster": ordinary, "example-cluster-acceptance": acceptance}},
    )

    acceptance_profile = resolve_cluster_profile("example-cluster-acceptance", config_path=config_path)
    ordinary_profile = resolve_cluster_profile("example-cluster", config_path=config_path)

    assert acceptance_profile.name == "example-cluster-acceptance"
    assert acceptance_profile.account == ordinary_profile.account == "example-account"
    assert acceptance_profile.resources["gpu_worker"].partition == "example-gpu"
    assert acceptance_profile.resources["gpu_worker"].nodelist == "gpu-node-017"
    assert ordinary_profile.resources["gpu_worker"].nodelist is None
    assert set(acceptance_profile.resources) == set(ordinary_profile.resources)
    for resource_name, ordinary_resource in ordinary_profile.resources.items():
        assert acceptance_profile.resources[resource_name].model_dump(
            exclude={"nodelist"}
        ) == ordinary_resource.model_dump(exclude={"nodelist"})
    assert "nodelist" not in ordinary_profile.resources["gpu_worker"].model_dump(exclude_none=True)
    assert acceptance_profile.database_acceptance_cache is not None
    assert ordinary_profile.database_acceptance_cache is None


@pytest.mark.parametrize(
    "nodelist",
    ["", "gpu-node-[001-016]", "gpu-node-017,gpu-node-018", "gpu-node-017\n#SBATCH --account=other", "$(id)"],
)
def test_cluster_profile_rejects_nonexact_or_unsafe_nodelist(tmp_path: Path, nodelist: str) -> None:
    profile = _profile_data()
    profile["transport"] = "local-slurm"
    profile["resources"] = {"gpu_worker": {"nodelist": nodelist}}
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": profile}})

    with pytest.raises(ValueError, match="Invalid resource override"):
        resolve_cluster_profile("example-cluster", config_path=config_path)


def test_resolve_cluster_profile_accepts_complete_custom_resource(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    config_path = _write_yaml(
        tmp_path / "profiles.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "owner": "tester",
                    "database_access_policies": ["direct"],
                    "project_root": "/project",
                    "output_root": "/output",
                    "staging_root": "/staging",
                    "afdb_toolkit_repo": "/toolkit",
                    "orchestration_repo": "/orchestration",
                    "image": "/images/bspp.sqsh",
                    "transport": "local-slurm",
                    "resources": {
                        "custom_check": {
                            "partition": "cpu",
                            "cpus_per_task": 2,
                            "memory": "8G",
                            "time": "00:10:00",
                        }
                    },
                }
            }
        },
    )

    profile = resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    assert profile.resources["custom_check"].partition == "cpu"


def test_resolve_cluster_profile_rejects_partial_custom_resource_with_named_error(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    config_path = _write_yaml(
        tmp_path / "profiles.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "owner": "tester",
                    "database_access_policies": ["direct"],
                    "project_root": "/project",
                    "output_root": "/output",
                    "staging_root": "/staging",
                    "afdb_toolkit_repo": "/toolkit",
                    "orchestration_repo": "/orchestration",
                    "image": "/images/bspp.sqsh",
                    "transport": "local-slurm",
                    "resources": {"custom_check": {"cpus_per_task": 2}},
                }
            }
        },
    )

    with pytest.raises(ValueError, match="custom_check"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_rejects_unknown_cluster(tmp_path: Path) -> None:
    template_path = _write_yaml(tmp_path / "templates.yaml", {"clusters": {}})
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {}})

    with pytest.raises(ValueError, match="missing from repo Cluster Profile Template, user Cluster Profile"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_non_example_typo_name_does_not_inherit_example_scheduler_authority(tmp_path: Path) -> None:
    """A typoed non-example name must fail closed instead of leaking example defaults."""
    data = _profile_data()
    data["transport"] = "local-slurm"
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"exampl-cluster": data}})

    with pytest.raises(ValueError, match="repo Cluster Profile Template"):
        resolve_cluster_profile("exampl-cluster", config_path=config_path)


def test_non_example_custom_cluster_resolves_only_authored_authority(tmp_path: Path) -> None:
    data = _profile_data()
    data.update(
        {
            "transport": "local-slurm",
            "account": "custom-account",
            "resources": {
                "gpu_worker": {
                    "partition": "custom-gpu",
                    "cpus_per_task": 8,
                    "memory": "64G",
                    "time": "01:00:00",
                    "gres": "gpu:1",
                },
                "analysis_finalize": {
                    "partition": "custom-cpu",
                    "cpus_per_task": 4,
                    "memory": "16G",
                    "time": "00:30:00",
                },
            },
        }
    )
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"custom-cluster": data}})

    profile = resolve_cluster_profile("custom-cluster", config_path=config_path)

    assert profile.name == "custom-cluster"
    assert profile.account == "custom-account"
    assert set(profile.resources) == {"gpu_worker", "analysis_finalize"}
    assert profile.resources["gpu_worker"].partition == "custom-gpu"
    assert profile.resources["analysis_finalize"].partition == "custom-cpu"


def test_resolve_cluster_profile_rejects_inheritance(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    config_path = _write_yaml(
        tmp_path / "profiles.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "extends": "base",
                    "owner": "tester",
                    "project_root": "/project",
                    "output_root": "/output",
                    "staging_root": "/staging",
                    "afdb_toolkit_repo": "/toolkit",
                    "orchestration_repo": "/orchestration",
                    "image": "/images/bspp.sqsh",
                    "transport": "local-slurm",
                }
            }
        },
    )

    with pytest.raises(ValueError, match="inheritance"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


@pytest.mark.parametrize("source", ["template", "user"])
def test_resolve_cluster_profile_rejects_environment_interpolation(tmp_path: Path, source: str) -> None:
    template = {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}}
    user = {
        "clusters": {
            "example-cluster": {
                "owner": "tester",
                "project_root": "/project",
                "output_root": "/output",
                "staging_root": "/staging",
                "afdb_toolkit_repo": "/toolkit",
                "orchestration_repo": "/orchestration",
                "image": "/images/bspp.sqsh",
                "transport": "local-slurm",
            }
        }
    }
    if source == "template":
        template["clusters"]["example-cluster"]["account"] = "$ACCOUNT"
    else:
        user["clusters"]["example-cluster"]["owner"] = "${USER}"

    template_path = _write_yaml(tmp_path / "templates.yaml", template)
    config_path = _write_yaml(tmp_path / "profiles.yaml", user)

    with pytest.raises(ValueError, match="environment interpolation"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_requires_explicit_transport(tmp_path: Path) -> None:
    template_path = _write_yaml(tmp_path / "templates.yaml", {"clusters": {"example-cluster": {"account": "acct"}}})
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": _profile_data()}})

    with pytest.raises(ValueError, match="transport"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_requires_ssh_target_for_ssh(tmp_path: Path) -> None:
    template_path = _write_yaml(tmp_path / "templates.yaml", {"clusters": {"example-cluster": {"account": "acct"}}})
    data = _profile_data()
    data["transport"] = "ssh"
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    with pytest.raises(ValueError, match="ssh_target"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_rejects_ssh_target_for_local_slurm(tmp_path: Path) -> None:
    template_path = _write_yaml(tmp_path / "templates.yaml", {"clusters": {"example-cluster": {"account": "acct"}}})
    data = _profile_data()
    data["transport"] = "local-slurm"
    data["ssh_target"] = "login"
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    with pytest.raises(ValueError, match="ssh_target"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_rejects_slurmrestd_transport(tmp_path: Path) -> None:
    template_path = _write_yaml(tmp_path / "templates.yaml", {"clusters": {"example-cluster": {"account": "acct"}}})
    data = _profile_data()
    data["transport"] = "slurmrestd"
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    with pytest.raises(ValueError, match="transport"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_database_profile_defaults_to_stage_required_and_resolves_exact_identity(tmp_path: Path) -> None:
    manifest_path = tmp_path / "database-source-manifest.json"
    data = _profile_data()
    data.pop("database_access_policies")
    data.update(
        {
            "transport": "local-slurm",
            "database_sets": [
                {
                    "identifier": "bspp-search",
                    "version": "2026-08",
                    "manifest_path": str(manifest_path),
                }
            ],
            "database_cache_root": "/node-cache/bspp",
            "database_cache_unix_user": "tester",
            "database_cache_filesystem_type": "ext4",
            "database_cache_reserve_bytes": 0,
            "database_lock_wait_seconds": 3600,
        }
    )
    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path, time="01:00:00"),
    )

    selection = DatabaseSetSelection(database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"))
    assert profile.database_access_policies == (DatabaseAccessPolicy.STAGE_REQUIRED,)
    assert profile.database_staging is not None
    assert profile.database_staging.lock_wait_seconds == 3600
    assert resolve_database_manifest_path(profile, selection) == manifest_path


def test_current_main_profile_without_database_configuration_remains_valid(tmp_path: Path) -> None:
    data = _profile_data()
    data.pop("database_access_policies")
    data["transport"] = "local-slurm"

    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path),
    )

    assert profile.database_access_policies == (DatabaseAccessPolicy.STAGE_REQUIRED,)
    assert profile.database_staging is None


def test_explicit_stage_required_profile_rejects_omitted_staging_authority(tmp_path: Path) -> None:
    data = _profile_data()
    data["database_access_policies"] = ["stage-required"]
    data["transport"] = "local-slurm"

    required_fields = (
        "database_cache_root, database_cache_unix_user, database_cache_filesystem_type, "
        "database_cache_reserve_bytes, database_lock_wait_seconds"
    )
    with pytest.raises(ValueError, match=required_fields):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path),
        )


def test_database_mapping_with_default_stage_required_rejects_omitted_staging_authority(tmp_path: Path) -> None:
    data = _profile_data()
    data.pop("database_access_policies")
    data["transport"] = "local-slurm"
    data["database_sets"] = [
        {
            "identifier": "bspp-search",
            "version": "2026-08",
            "manifest_path": str(tmp_path / "manifest.json"),
        }
    ]

    with pytest.raises(ValueError, match="requires all database cache fields"):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path),
        )


def test_current_main_profile_cannot_resolve_preprocessing_database_selection(tmp_path: Path) -> None:
    data = _profile_data()
    data.pop("database_access_policies")
    data["transport"] = "local-slurm"
    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path),
    )
    selection = DatabaseSetSelection(database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"))

    with pytest.raises(ValueError, match="requires complete database staging authority"):
        resolve_database_manifest_path(profile, selection)


def test_database_profile_freezes_explicit_cache_unix_user_not_evidence_owner(tmp_path: Path) -> None:
    data = _staging_database_profile_data(tmp_path)
    data["owner"] = "evidence-owner"
    data["database_cache_unix_user"] = "batch-user"

    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path),
    )

    assert profile.owner == "evidence-owner"
    assert profile.database_staging is not None
    assert profile.database_staging.unix_user == "batch-user"


def test_staging_database_profile_requires_explicit_cache_unix_user(tmp_path: Path) -> None:
    data = _staging_database_profile_data(tmp_path)
    data.pop("database_cache_unix_user")

    with pytest.raises(ValueError, match="database_cache_unix_user"):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path),
        )


def test_staging_database_profile_rejects_unsafe_cache_unix_user(tmp_path: Path) -> None:
    data = _staging_database_profile_data(tmp_path)
    data["database_cache_unix_user"] = "../other"

    with pytest.raises(ValueError, match="safe Unix user identity"):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path),
        )


def test_database_profile_denies_a_requested_policy_outside_its_allowlist(tmp_path: Path) -> None:
    manifest_path = tmp_path / "database-source-manifest.json"
    data = _profile_data()
    data.update(
        {
            "transport": "local-slurm",
            "database_access_policies": ["stage-required"],
            "database_sets": [
                {
                    "identifier": "bspp-search",
                    "version": "2026-08",
                    "manifest_path": str(manifest_path),
                }
            ],
            "database_cache_root": "/node-cache/bspp",
            "database_cache_unix_user": "tester",
            "database_cache_filesystem_type": "ext4",
            "database_cache_reserve_bytes": 0,
            "database_lock_wait_seconds": 3600,
        }
    )
    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path, time="01:00:00"),
    )
    selection = DatabaseSetSelection(
        database_set=DatabaseSetIdentity(identifier="bspp-search", version="2026-08"),
        requested_policy=DatabaseAccessPolicy.DIRECT,
    )

    with pytest.raises(ValueError, match="does not permit database policy 'direct'"):
        resolve_database_manifest_path(profile, selection)


def test_direct_only_database_profile_may_omit_staging_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "database-source-manifest.json"
    data = _profile_data()
    data.update(
        {
            "transport": "local-slurm",
            "database_access_policies": ["direct"],
            "database_sets": [
                {
                    "identifier": "bspp-search",
                    "version": "2026-08",
                    "manifest_path": str(manifest_path),
                }
            ],
        }
    )
    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path),
    )

    assert profile.database_access_policies == (DatabaseAccessPolicy.DIRECT,)
    assert profile.database_staging is None


def test_direct_only_database_profile_accepts_complete_staging_authority(tmp_path: Path) -> None:
    data = _staging_database_profile_data(tmp_path)
    data["database_access_policies"] = ["direct"]

    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path),
    )

    assert profile.database_access_policies == (DatabaseAccessPolicy.DIRECT,)
    assert profile.database_staging is not None
    assert profile.database_staging.unix_user == "tester"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"database_cache_root": None}, "requires all database cache fields"),
        ({"database_lock_wait_seconds": 3601}, "cannot exceed"),
        ({"database_sets": [{"identifier": "x", "version": "v", "manifest_path": "relative.json"}]}, "absolute"),
    ],
)
def test_database_profile_rejects_incomplete_overlong_or_relative_site_authority(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    data = _profile_data()
    data.update(
        {
            "transport": "local-slurm",
            "database_access_policies": ["stage-required"],
            "database_sets": [
                {
                    "identifier": "bspp-search",
                    "version": "2026-08",
                    "manifest_path": str(tmp_path / "manifest.json"),
                }
            ],
            "database_cache_root": "/node-cache/bspp",
            "database_cache_unix_user": "tester",
            "database_cache_filesystem_type": "ext4",
            "database_cache_reserve_bytes": 0,
            "database_lock_wait_seconds": 3600,
        }
    )
    data.update(mutation)

    with pytest.raises(ValueError, match=message):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path, time="01:00:00"),
        )


def test_database_profile_rejects_negative_cache_reserve(tmp_path: Path) -> None:
    data = _staging_database_profile_data(tmp_path)
    data["database_cache_reserve_bytes"] = -1

    with pytest.raises(ValueError, match="database_cache_reserve_bytes must be non-negative"):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path),
        )


@pytest.mark.parametrize("lock_wait_seconds", [0, -1])
def test_database_profile_rejects_nonpositive_lock_wait(
    tmp_path: Path,
    lock_wait_seconds: int,
) -> None:
    data = _staging_database_profile_data(tmp_path)
    data["database_lock_wait_seconds"] = lock_wait_seconds

    with pytest.raises(ValueError, match="database_lock_wait_seconds must be positive"):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path),
        )


def test_database_profile_rejects_duplicate_database_set_identity_keys(tmp_path: Path) -> None:
    data = _staging_database_profile_data(tmp_path)
    data["database_sets"] = [
        {
            "identifier": "bspp-search",
            "version": "2026-08",
            "manifest_path": str(tmp_path / "first.json"),
        },
        {
            "identifier": "bspp-search",
            "version": "2026-08",
            "manifest_path": str(tmp_path / "second.json"),
        },
    ]

    with pytest.raises(ValueError, match="database_sets must not contain duplicate identities"):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path),
        )


def test_database_profile_rejects_unknown_database_access_policy(tmp_path: Path) -> None:
    data = _staging_database_profile_data(tmp_path)
    data["database_access_policies"] = ["stage-sometimes"]

    with pytest.raises(ValueError, match="stage-sometimes"):
        resolve_cluster_profile(
            "example-cluster",
            config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
            template_path=_database_template(tmp_path),
        )


def _database_template(tmp_path: Path, *, time: str = "1-00:00:00") -> Path:
    return _write_yaml(
        tmp_path / "templates.yaml",
        {
            "clusters": {
                "example-cluster": {
                    "account": "acct",
                    "resources": {
                        "gpu_worker": {
                            "partition": "gpu",
                            "cpus_per_task": 8,
                            "memory": "32G",
                            "time": time,
                            "gres": "gpu:1",
                        }
                    },
                }
            }
        },
    )


def _profile_data() -> dict[str, object]:
    return {
        "owner": "tester",
        "database_access_policies": ["direct"],
        "paths": {
            "project_root": "/project",
            "output_root": "/output",
            "staging_root": "/staging",
            "afdb_toolkit_repo": "/toolkit",
            "orchestration_repo": "/orchestration",
            "image": "/images/bspp.sqsh",
        },
    }


def _staging_database_profile_data(tmp_path: Path) -> dict[str, object]:
    data = _profile_data()
    data.update(
        {
            "transport": "local-slurm",
            "database_access_policies": ["stage-required"],
            "database_sets": [
                {
                    "identifier": "bspp-search",
                    "version": "2026-08",
                    "manifest_path": str(tmp_path / "manifest.json"),
                }
            ],
            "database_cache_root": "/node-cache/bspp",
            "database_cache_unix_user": "tester",
            "database_cache_filesystem_type": "ext4",
            "database_cache_reserve_bytes": 0,
            "database_lock_wait_seconds": 3600,
        }
    )
    return data


def test_resolve_cluster_profile_without_toolkit_accepts_and_resolves_to_none(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    data = _profile_data()
    paths = data["paths"]
    assert isinstance(paths, dict)
    paths.pop("afdb_toolkit_repo")
    data["transport"] = "local-slurm"
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    profile = resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    assert profile.afdb_toolkit_repo is None
    assert profile.orchestration_repo == "/orchestration"
    assert profile.project_root == "/project"


def test_resolve_cluster_profile_with_explicit_toolkit_preserves_behavior(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    data = _profile_data()
    data["transport"] = "local-slurm"
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    profile = resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    assert profile.afdb_toolkit_repo == "/toolkit"
    assert profile.orchestration_repo == "/orchestration"


def test_ssh_baked_profile_preserves_governed_and_preprocessing_database_authority(tmp_path: Path) -> None:
    data = _staging_database_profile_data(tmp_path)
    paths = data["paths"]
    assert isinstance(paths, dict)
    paths.pop("afdb_toolkit_repo")
    paths.update(
        {
            "source_bundle_root": "/cluster/source-bundles",
            "runtime_qualification_root": "/cluster/runtime-qualifications",
            "runtime_qualification_control_root": "/workstation/runtime-qualifications",
        }
    )
    data.update(
        {
            "transport": "ssh",
            "ssh_target": "example-cluster-login",
            "preprocessing_runtime": {
                "cluster_image_path": "/images/preprocessing.sqsh",
                "cluster_image_sha256": "a" * 64,
                "oci_digest": "sha256:" + "b" * 64,
                "image_lock_sha256": "c" * 64,
                "contract_wheel_sha256": "d" * 64,
                "runtime_wheel_sha256": "e" * 64,
                "control_wheel_sha256": "f" * 64,
                "source_commit": "f" * 40,
                "source_bundle_sha256": "1" * 64,
                "colabfold_version": "1.6.2",
                "mmseqs_version": "18-8cc5c",
                "rsync_version": "3.4.4",
                "cuda_version": "12.6.3",
            },
        }
    )

    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path),
    )

    assert profile.transport == "ssh"
    assert profile.afdb_toolkit_repo is None
    assert profile.governed_package_root == "/cluster/runtime-qualifications/packages"
    assert profile.runtime_qualification_root == "/cluster/runtime-qualifications"
    assert profile.runtime_qualification_control_root == "/workstation/runtime-qualifications"
    assert profile.preprocessing_runtime is not None
    assert profile.database_staging is not None
    assert profile.database_access_policies == (DatabaseAccessPolicy.STAGE_REQUIRED,)


def test_legacy_preprocessing_profile_without_control_wheel_sha256_resolves(tmp_path: Path) -> None:
    """A pre-Control 12-field preprocessing profile still loads; the Control
    wheel digest is optional and resolves to None."""
    data = _profile_data()
    data.update(
        {
            "transport": "ssh",
            "ssh_target": "example-cluster-login",
            "preprocessing_runtime": {
                "cluster_image_path": "/images/preprocessing.sqsh",
                "cluster_image_sha256": "a" * 64,
                "oci_digest": "sha256:" + "b" * 64,
                "image_lock_sha256": "c" * 64,
                "contract_wheel_sha256": "d" * 64,
                "runtime_wheel_sha256": "e" * 64,
                "source_commit": "f" * 40,
                "source_bundle_sha256": "1" * 64,
                "colabfold_version": "1.6.2",
                "mmseqs_version": "18-8cc5c",
                "rsync_version": "3.4.4",
                "cuda_version": "12.6.3",
            },
        }
    )

    profile = resolve_cluster_profile(
        "example-cluster",
        config_path=_write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}}),
        template_path=_database_template(tmp_path),
    )

    assert profile.preprocessing_runtime is not None
    assert profile.preprocessing_runtime.control_wheel_sha256 is None


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# --- typed packed-topology resource tests (e13s02) ---


def test_resolve_cluster_profile_parses_typed_topology(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    data = _profile_data()
    data["transport"] = "local-slurm"
    data["resources"] = {
        "gpu_worker": {
            "partition": "gpu",
            "cpus_per_task": 8,
            "memory": "64G",
            "time": "01:00:00",
            "nodes": 2,
            "tasks_per_node": 8,
            "gpus_per_task": 1,
            "max_parallel": 2,
        }
    }
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    profile = resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)

    resource = profile.resources["gpu_worker"]
    assert resource.nodes == 2
    assert resource.tasks_per_node == 8
    assert resource.gpus_per_task == 1
    assert resource.max_parallel == 2
    dumped = resource.model_dump()
    assert dumped["nodes"] == 2
    assert dumped["tasks_per_node"] == 8
    assert dumped["gpus_per_task"] == 1
    assert dumped["max_parallel"] == 2


@pytest.mark.parametrize("conflict", ["array", "gres"])
def test_resolve_cluster_profile_rejects_conflicting_legacy_topology(
    tmp_path: Path,
    conflict: str,
) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    data = _profile_data()
    data["transport"] = "local-slurm"
    resource = {
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "time": "01:00:00",
        "nodes": 2,
        "tasks_per_node": 8,
        "gpus_per_task": 1,
        "max_parallel": 2,
    }
    if conflict == "array":
        resource["array"] = "0-3"
    else:
        resource["gres"] = "gpu:1"
    data["resources"] = {"gpu_worker": resource}
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    with pytest.raises(ValueError, match="legacy"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"tasks_per_node": 4}, "tasks_per_node requires typed topology"),
        ({"gpus_per_task": 1}, "gpus_per_task requires typed topology"),
        ({"max_parallel": 2}, "max_parallel requires typed topology"),
    ],
)
def test_resolve_cluster_profile_rejects_incomplete_typed_topology(
    tmp_path: Path,
    extra: dict[str, int],
    match: str,
) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    data = _profile_data()
    data["transport"] = "local-slurm"
    resource = {
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "time": "01:00:00",
        **extra,
    }
    data["resources"] = {"gpu_worker": resource}
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    with pytest.raises(ValueError, match=match):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_rejects_typed_topology_without_gpus_per_task(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    data = _profile_data()
    data["transport"] = "local-slurm"
    data["resources"] = {
        "gpu_worker": {
            "partition": "gpu",
            "cpus_per_task": 8,
            "memory": "64G",
            "time": "01:00:00",
            "nodes": 2,
            "tasks_per_node": 8,
        }
    }
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    with pytest.raises(ValueError, match="requires gpus_per_task"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)


def test_resolve_cluster_profile_rejects_degenerate_typed_one_worker_topology(tmp_path: Path) -> None:
    template_path = _write_yaml(
        tmp_path / "templates.yaml",
        {"clusters": {"example-cluster": {"account": "acct", "resources": {}}}},
    )
    data = _profile_data()
    data["transport"] = "local-slurm"
    data["resources"] = {
        "gpu_worker": {
            "partition": "gpu",
            "cpus_per_task": 8,
            "memory": "64G",
            "time": "01:00:00",
            "nodes": 1,
            "tasks_per_node": 1,
            "gpus_per_task": 1,
        }
    }
    config_path = _write_yaml(tmp_path / "profiles.yaml", {"clusters": {"example-cluster": data}})

    with pytest.raises(ValueError, match="degenerate one-worker shape"):
        resolve_cluster_profile("example-cluster", config_path=config_path, template_path=template_path)
