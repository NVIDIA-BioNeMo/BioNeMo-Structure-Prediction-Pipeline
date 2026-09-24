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

"""Configuration loading from TOML and YAML files."""

from __future__ import annotations

import importlib.resources
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

__all__ = [
    "ClusterDefinition",
    "ClustersConfig",
    "GCSConfig",
    "GCSPathsConfig",
    "LocalConfig",
    "PipelinesConfig",
    "S3Config",
    "load_cluster_config",
    "load_cluster_config_model",
    "load_config",
    "load_pipelines_config",
    "load_yaml",
    "resolve_path",
]


class GCSPathsConfig(BaseModel):
    """Optional GCS object paths from ``pipelines.toml``."""

    model_config = ConfigDict(extra="allow")

    manifest: str | None = None


class GCSConfig(BaseModel):
    """GCS transfer settings from ``pipelines.toml``."""

    model_config = ConfigDict(extra="allow")

    credentials: str
    bucket: str
    prefix: str = ""
    paths: GCSPathsConfig | None = None


class S3Config(BaseModel):
    """S3 transfer settings from ``pipelines.toml``."""

    model_config = ConfigDict(extra="allow")

    bucket: str
    prefix: str = ""
    endpoint_url_env: str = "S3_ENDPOINT_URL"


class LocalConfig(BaseModel):
    """Optional local path settings from ``pipelines.toml``."""

    model_config = ConfigDict(extra="allow")

    base_dir: str = ""


class PipelinesConfig(BaseModel):
    """Typed representation of ``pipelines.toml``.

    Extra top-level sections are preserved so callers can add operational
    configuration without needing a library release first.
    """

    model_config = ConfigDict(extra="allow")

    gcs: GCSConfig | None = None
    s3: S3Config | None = None
    local: LocalConfig | None = None

    def to_compat_dict(self) -> dict[str, Any]:
        """Return a dict shaped like ``tomllib.load`` output."""
        return self.model_dump(exclude_none=True, exclude_unset=True)


class ClusterDefinition(BaseModel):
    """Typed YAML definition for one cluster entry."""

    model_config = ConfigDict(extra="allow")

    filesystem: str
    partition_gpu: str
    partition_cpu: str
    job_reaper_comment: bool = False


class ClustersConfig(BaseModel):
    """Typed representation of bundled or custom ``clusters.yaml``."""

    model_config = ConfigDict(extra="allow")

    clusters: dict[str, ClusterDefinition]

    def to_compat_dict(self) -> dict[str, Any]:
        """Return a plain dict compatible with the historical YAML loader."""
        return self.model_dump(exclude_none=True, exclude_unset=True)


def load_config(path: Path) -> dict[str, Any]:
    """Load a TOML configuration file.

    Credential paths in the config are resolved relative to the config
    file's parent directory.
    """
    with path.open("rb") as f:
        return tomllib.load(f)


def load_pipelines_config(path: Path) -> PipelinesConfig:
    """Load and validate ``pipelines.toml`` as a typed Pydantic model."""
    return PipelinesConfig.model_validate(load_config(path))


def resolve_path(config_path: Path, relative: str) -> Path:
    """Resolve a path from the config relative to the config file's directory."""
    return (config_path.parent / relative).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with path.open() as f:
        result = yaml.safe_load(f)
    if result is None:
        return {}
    if not isinstance(result, dict):
        msg = f"Expected a YAML mapping, got {type(result).__name__}"
        raise TypeError(msg)
    return result


_NVIDIA_OVERLAY_FILENAME = "nvidia.toml"


def _merge_nvidia_overlay(data: dict[str, Any]) -> dict[str, Any]:
    """Merge the repo-root ``nvidia.toml`` overlay over bundled cluster data.

    The overlay is discovered relative to the process working directory
    (matching the cwd-relative ``pipelines.toml`` CLI default). When it is not
    present, *data* is returned unchanged. Only the overlay's ``clusters``
    mapping is merged key-by-key so the generic ``example`` entry survives
    alongside operator entries. Unrelated top-level keys (for example
    environment-specific container pins owned outside this file) are ignored
    rather than passed into :class:`ClustersConfig`.
    """
    overlay_path = Path(_NVIDIA_OVERLAY_FILENAME)
    if not overlay_path.is_file():
        return data
    overlay = load_config(overlay_path)
    merged = dict(data)
    clusters = merged.get("clusters")
    overlay_clusters = overlay.get("clusters")
    if isinstance(clusters, Mapping) and isinstance(overlay_clusters, Mapping):
        merged["clusters"] = {**clusters, **overlay_clusters}
    return merged


def load_cluster_config(path: Path | None = None) -> dict[str, Any]:
    """Load cluster configuration from YAML.

    If *path* is ``None``, loads the bundled ``clusters.yaml`` shipped with
    the package via :mod:`importlib.resources` and merges the repo-root
    ``nvidia.toml`` overlay when present.
    """
    if path is not None:
        return load_yaml(path)
    ref = importlib.resources.files("bspp.orchestration.runtime.data").joinpath("clusters.yaml")
    with importlib.resources.as_file(ref) as p:
        data = load_yaml(p)
    return _merge_nvidia_overlay(data)


def load_cluster_config_model(path: Path | None = None) -> ClustersConfig:
    """Load and validate cluster configuration as a typed model."""
    return ClustersConfig.model_validate(load_cluster_config(path))
