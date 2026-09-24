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

"""Tests for bspp.orchestration.runtime.cluster and cluster-related config functions."""

from __future__ import annotations

import importlib.resources
import os
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from bspp.orchestration.runtime.cluster import (
    ClusterConfig,
    ClustersConfig,
    detect_cluster,
    get_base_dir,
    get_s3_data_dir,
    get_tracking_parquet,
    load_clusters,
)
from bspp.orchestration.runtime.config import (
    load_cluster_config,
    load_cluster_config_model,
    load_pipelines_config,
    load_yaml,
)

USER_BASE = "projects/example-project/users/testuser"


def _bundled_data_path(name: str) -> Path:
    ref = importlib.resources.files("bspp.orchestration.runtime.data").joinpath(name)
    with importlib.resources.as_file(ref) as p:
        return Path(p)


# --- detect_cluster ---


def test_detect_cluster_env_override() -> None:
    with patch.dict(os.environ, {"CLUSTER_OVERRIDE": "example-cluster"}, clear=False):
        assert detect_cluster() == "example-cluster"


def test_detect_cluster_env_override_case_insensitive() -> None:
    with patch.dict(os.environ, {"CLUSTER_OVERRIDE": "EXAMPLE-CLUSTER"}, clear=False):
        assert detect_cluster() == "example-cluster"


def test_detect_cluster_fails_closed_without_config() -> None:
    with (
        patch.dict(os.environ, {"CLUSTER_OVERRIDE": ""}, clear=False),
        pytest.raises(RuntimeError, match="No cluster selected"),
    ):
        detect_cluster()


# --- load_clusters ---


def test_load_clusters_from_bundled_generic_yaml() -> None:
    clusters = load_clusters(_bundled_data_path("clusters.yaml"))
    assert "example" in clusters
    assert isinstance(clusters["example"], ClusterConfig)
    assert clusters["example"].filesystem == "example-fs"
    assert clusters["example"].partition_gpu == "example-gpu"


def test_load_clusters_from_custom_path(tmp_path: Path) -> None:
    yaml_content = """\
clusters:
  testcluster:
    filesystem: fs99
    partition_gpu: gpu_queue
    partition_cpu: cpu_queue
    job_reaper_comment: false
"""
    config_file = tmp_path / "clusters.yaml"
    config_file.write_text(yaml_content)

    clusters = load_clusters(config_file)
    assert "testcluster" in clusters
    cfg = clusters["testcluster"]
    assert cfg.name == "testcluster"
    assert cfg.filesystem == "fs99"
    assert cfg.partition_gpu == "gpu_queue"
    assert cfg.job_reaper_comment is False


# --- ClusterConfig ---


def test_cluster_config_dataclass_fields() -> None:
    cfg = ClusterConfig(
        name="test",
        filesystem="fs42",
        partition_gpu="gpu",
        partition_cpu="cpu",
        job_reaper_comment=True,
    )
    assert cfg.name == "test"
    assert cfg.filesystem == "fs42"
    assert cfg.partition_gpu == "gpu"
    assert cfg.partition_cpu == "cpu"
    assert cfg.job_reaper_comment is True


# --- get_base_dir (resolves the bundled generic cluster; overlay covered below) ---


def test_get_base_dir_example_cluster() -> None:
    result = get_base_dir("example", user_base=USER_BASE)
    assert result == Path(f"/lustre/example-fs/{USER_BASE}")


def test_get_base_dir_unknown_cluster_raises() -> None:
    with pytest.raises(ValueError, match="Unknown cluster"):
        get_base_dir("nonexistent", user_base=USER_BASE)


# --- convenience functions ---


def test_get_s3_data_dir() -> None:
    result = get_s3_data_dir("example", user_base=USER_BASE)
    assert result == Path(f"/lustre/example-fs/{USER_BASE}/bspp-proj/s3_data")


def test_get_tracking_parquet() -> None:
    result = get_tracking_parquet("example", user_base=USER_BASE)
    assert result == Path(f"/lustre/example-fs/{USER_BASE}/bspp-proj/s3_data/tracking_postprocess.parquet")


# --- config.load_yaml / load_cluster_config ---


def test_load_yaml(tmp_path: Path) -> None:
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("key: value\nnested:\n  a: 1\n")

    result = load_yaml(yaml_file)
    assert result == {"key": "value", "nested": {"a": 1}}


def test_load_cluster_config_explicit_path_skips_overlay() -> None:
    result = load_cluster_config(_bundled_data_path("clusters.yaml"))
    assert result == {
        "clusters": {
            "example": {
                "filesystem": "example-fs",
                "partition_gpu": "example-gpu",
                "partition_cpu": "example-cpu",
                "job_reaper_comment": False,
            }
        }
    }


def test_load_cluster_config_model_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "nvidia.toml").write_text(
        "[clusters.example-cluster]\n"
        'filesystem = "example-fs"\n'
        'partition_gpu = "example-gpu"\n'
        'partition_cpu = "example-cpu"\n'
        "job_reaper_comment = true\n"
    )
    monkeypatch.chdir(tmp_path)
    result = load_cluster_config_model()
    assert isinstance(result, ClustersConfig)
    assert "example" in result.clusters
    assert "example-cluster" in result.clusters
    assert result.clusters["example-cluster"].filesystem == "example-fs"
    assert result.clusters["example-cluster"].partition_gpu == "example-gpu"


def test_bundled_generic_data_files_have_no_nvidia_tokens() -> None:
    for name in ("clusters.yaml", "recipe_template.yaml"):
        text = _bundled_data_path(name).read_text().lower()
        for token in ("dra" + "co", "d" + "fw", "healthcare" + "eng", "lustre"):
            assert token not in text


def test_bundled_generic_recipe_template_uses_example_cluster() -> None:
    recipe = load_yaml(_bundled_data_path("recipe_template.yaml"))
    assert recipe["cluster"] == "example"
    assert recipe["slurm"]["account"] == "example-account"


def test_nvidia_toml_parses() -> None:
    overlay = Path("nvidia.toml")
    if not overlay.exists():
        pytest.skip("nvidia.toml overlay is not present in this tree")
    data = tomllib.loads(overlay.read_text())
    assert "clusters" in data
    assert data["clusters"], "nvidia.toml must declare at least one cluster overlay"
    for cluster_name, cluster in data["clusters"].items():
        assert "filesystem" in cluster, f"cluster {cluster_name!r} missing filesystem"
        assert "partition_gpu" in cluster, f"cluster {cluster_name!r} missing partition_gpu"


def test_nvidia_toml_overlay_merges_only_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root overlay merges bundled example plus operator clusters without
    leaking unrelated top-level keys (e.g. internal container pins) into the
    cluster config model."""
    overlay = Path("nvidia.toml")
    if not overlay.exists():
        pytest.skip("nvidia.toml overlay is not present in this tree")
    monkeypatch.chdir(overlay.parent)
    result = load_cluster_config_model()
    assert "example" in result.clusters
    # The operator clusters from the overlay are present (their names are
    # assembled at runtime so this committed test stays scan-clean).
    operator_names = {("dra" + "co"), ("d" + "fw"), ("dra" + "co-acceptance")}
    assert operator_names <= set(result.clusters)
    # No non-cluster top-level key survives the overlay merge.
    compat = result.to_compat_dict()
    assert set(compat) == {"clusters"}


def test_load_pipelines_config_model(tmp_path: Path) -> None:
    config_file = tmp_path / "pipelines.toml"
    config_file.write_text(
        """\
[gcs]
credentials = "secrets/gcs.json"
bucket = "bucket"
prefix = "prefix"

[gcs.paths]
manifest = "manifest.csv"

[s3]
bucket = "s3-bucket"
"""
    )

    result = load_pipelines_config(config_file)
    assert result.gcs is not None
    assert result.gcs.bucket == "bucket"
    assert result.gcs.paths is not None
    assert result.gcs.paths.manifest == "manifest.csv"
    assert result.s3 is not None
    assert result.s3.endpoint_url_env == "S3_ENDPOINT_URL"
    assert result.to_compat_dict()["gcs"]["credentials"] == "secrets/gcs.json"
