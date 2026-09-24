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

"""Cluster path resolution via explicit operator configuration.

Uses the bundled generic ``clusters.yaml`` for filesystem mapping, optionally
overlaid by an operator-supplied overlay file when present. Cluster selection is explicit only
(``CLUSTER_OVERRIDE``); no hostname inference or implicit default is applied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.runtime.config import ClusterDefinition, ClustersConfig, load_cluster_config_model

__all__ = [
    "ClusterConfig",
    "ClusterDefinition",
    "ClustersConfig",
    "detect_cluster",
    "get_base_dir",
    "get_input_datasets_dir",
    "get_master_parquet",
    "get_s3_data_dir",
    "get_tracking_parquet",
    "load_clusters",
]


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    """Configuration for a single HPC cluster."""

    name: str
    filesystem: str
    partition_gpu: str
    partition_cpu: str
    job_reaper_comment: bool


def detect_cluster() -> str:
    """Return the explicitly configured cluster name.

    Cluster selection is explicit only: ``CLUSTER_OVERRIDE`` names the
    cluster. No hostname inference and no implicit default are applied, so
    callers fail closed when the operator has not selected a cluster.
    """
    override = os.environ.get("CLUSTER_OVERRIDE", "").strip().lower()
    if override:
        return override
    raise RuntimeError(
        "No cluster selected: set CLUSTER_OVERRIDE to an explicit cluster name "
        "(or pass an explicit cluster name). No implicit cluster default is applied."
    )


def load_clusters(config_path: Path | None = None) -> dict[str, ClusterConfig]:
    """Load cluster definitions from YAML and return a name-keyed dict.

    When *config_path* is ``None`` the bundled ``clusters.yaml`` is used.
    """
    raw = load_cluster_config_model(config_path)
    result: dict[str, ClusterConfig] = {}
    for name, vals in raw.clusters.items():
        result[name] = ClusterConfig(
            name=name,
            filesystem=vals.filesystem,
            partition_gpu=vals.partition_gpu,
            partition_cpu=vals.partition_cpu,
            job_reaper_comment=vals.job_reaper_comment,
        )
    return result


def get_base_dir(
    cluster: str | None = None,
    *,
    user_base: str,
    config_path: Path | None = None,
) -> Path:
    """Return the filesystem base path for a cluster.

    Example return value::

        /lustre/<filesystem>/<user_base>

    Parameters
    ----------
    cluster:
        Explicit cluster name. Auto-detected from ``CLUSTER_OVERRIDE`` if
        ``None``.
    user_base:
        Required. The user-specific path segment under the filesystem mount,
        e.g. ``"projects/.../users/myuser"``.
    config_path:
        Optional path to a custom ``clusters.yaml``.
    """
    cluster = cluster or detect_cluster()
    clusters = load_clusters(config_path)
    cfg = clusters.get(cluster)
    if cfg is None:
        msg = f"Unknown cluster: {cluster!r}. Known clusters: {sorted(clusters)}"
        raise ValueError(msg)
    return Path(f"/lustre/{cfg.filesystem}/{user_base}")


def get_s3_data_dir(
    cluster: str | None = None,
    *,
    user_base: str,
    config_path: Path | None = None,
) -> Path:
    """Return the ``s3_data`` directory path."""
    return get_base_dir(cluster, user_base=user_base, config_path=config_path) / "bspp-proj/s3_data"


def get_input_datasets_dir(
    cluster: str | None = None,
    *,
    user_base: str,
    config_path: Path | None = None,
) -> Path:
    """Return the ``input_datasets`` base directory path."""
    return get_base_dir(cluster, user_base=user_base, config_path=config_path) / "bspp-proj/input_datasets"


def get_tracking_parquet(
    cluster: str | None = None,
    *,
    user_base: str,
    config_path: Path | None = None,
) -> Path:
    """Return the ``tracking_postprocess.parquet`` path."""
    return get_s3_data_dir(cluster, user_base=user_base, config_path=config_path) / "tracking_postprocess.parquet"


def get_master_parquet(
    cluster: str | None = None,
    *,
    user_base: str,
    config_path: Path | None = None,
) -> Path:
    """Return the ``master_all_runs_merged_23M.parquet`` path."""
    return get_s3_data_dir(cluster, user_base=user_base, config_path=config_path) / "master_all_runs_merged_23M.parquet"
