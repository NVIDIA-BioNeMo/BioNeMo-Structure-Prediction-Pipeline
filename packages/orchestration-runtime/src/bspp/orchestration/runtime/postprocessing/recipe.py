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

"""Recipe handling: generate and load recipe directories.

A recipe is a directory containing a ``config.yaml`` that configures a
SLURM pipeline run for one dataset.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

__all__ = [
    "RecipeConfig",
    "RecipeEnvironmentConfig",
    "RecipeMonitoringConfig",
    "RecipePathsConfig",
    "RecipeSlurmConfig",
    "RecipeWorkerConfig",
    "generate_recipe",
    "load_recipe",
    "load_recipe_config",
]


class RecipeSlurmConfig(BaseModel):
    """Typed ``slurm`` section for recipe ``config.yaml`` files."""

    model_config = ConfigDict(extra="allow")

    partition: str | None = None
    account: str | None = None
    array_range: str | None = None
    cpus_per_task: int | None = None
    memory: str | None = None
    gres: str | None = None
    time: str | None = None


class RecipePathsConfig(BaseModel):
    """Typed ``paths`` section for recipe ``config.yaml`` files."""

    model_config = ConfigDict(extra="allow")

    input_dir: str | None = None
    output_dir: str | None = None
    log_dir: str | None = None


class RecipeWorkerConfig(BaseModel):
    """Typed ``worker`` section for recipe ``config.yaml`` files."""

    model_config = ConfigDict(extra="allow")

    stages: str | None = None
    workers: int | None = None
    tool_used: str | None = None
    heterodimers: bool | None = None
    clash_device: str | None = None
    clash_batch_size: int | None = None
    dssp_algorithm: str | None = None


class RecipeEnvironmentConfig(BaseModel):
    """Typed ``environment`` section for recipe ``config.yaml`` files."""

    model_config = ConfigDict(extra="allow")

    python_env: str | None = None
    modules: str | None = None


class RecipeMonitoringConfig(BaseModel):
    """Typed ``monitoring`` section for recipe ``config.yaml`` files."""

    model_config = ConfigDict(extra="allow")

    enabled: bool | None = None
    refresh_interval: int | None = None
    alert_threshold: float | None = None


class RecipeConfig(BaseModel):
    """Typed recipe config.

    The model includes the generated recipe sections and permits extra
    fields used by older fixture-style recipe files.
    """

    model_config = ConfigDict(extra="allow")

    cluster: str | None = None
    job_name: str | None = None
    run_name: str | None = None
    slurm: RecipeSlurmConfig | None = None
    paths: RecipePathsConfig | None = None
    worker: RecipeWorkerConfig | None = None
    environment: RecipeEnvironmentConfig | None = None
    monitoring: RecipeMonitoringConfig | None = None

    def to_compat_dict(self) -> dict[str, Any]:
        """Return a dict compatible with the historical recipe loader."""
        return self.model_dump(exclude_none=True, exclude_unset=True)


_DEFAULT_CONFIG: dict[str, Any] = {
    "cluster": "example",
    "job_name": "bspp_postprocess",
    "run_name": "bspp_run",
    "slurm": {
        "partition": None,
        "account": "example-account",
        "array_range": "0-0",
        "cpus_per_task": 30,
        "memory": "128G",
        "gres": "gpu:1",
        "time": "04:00:00",
    },
    "paths": {
        "input_dir": "",
        "output_dir": "",
        "log_dir": "",
    },
    "worker": {
        "stages": "ipsae dssp validation metadata_export modelcif_export",
        "workers": 24,
        "tool_used": "ColabFold v1.6.0 / AlphaFold-Multimer",
        "heterodimers": False,
        "clash_device": "cuda",
        "clash_batch_size": 128,
        "dssp_algorithm": "pydssp",
    },
    "environment": {
        "python_env": "",
        "modules": "",
    },
    "monitoring": {
        "enabled": True,
        "refresh_interval": 10,
        "alert_threshold": 5.0,
    },
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into a copy of *base*."""
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_template(template_path: Path | None) -> dict[str, Any]:
    """Load a recipe template from *template_path* or fall back to defaults.

    If *template_path* is ``None``, tries to load the bundled
    ``config.yaml`` from the package data. If that also does not exist,
    returns the hardcoded default config.
    """
    if template_path is not None:
        with template_path.open() as f:
            loaded = yaml.safe_load(f)
        if isinstance(loaded, dict):
            return loaded
        return dict(_DEFAULT_CONFIG)

    # Try bundled template
    try:
        ref = importlib.resources.files("bspp.orchestration.runtime.data").joinpath("recipe_template.yaml")
        with importlib.resources.as_file(ref) as p:
            if p.exists():
                with p.open() as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    return dict(_DEFAULT_CONFIG)


def generate_recipe(
    dataset: str,
    output_dir: Path,
    *,
    template_path: Path | None = None,
    **overrides: Any,
) -> Path:
    """Create a recipe directory with ``config.yaml`` for *dataset*.

    The recipe config is built by merging a template (from *template_path*
    or the built-in default) with the provided *overrides*. The ``dataset``
    and ``run_name`` keys are set automatically.

    Returns the path to the created recipe directory.
    """
    base = _load_template(template_path)
    base["run_name"] = dataset
    base["job_name"] = f"bspp_{dataset}"

    config = _deep_merge(base, overrides)

    recipe_dir = output_dir / dataset
    recipe_dir.mkdir(parents=True, exist_ok=True)

    config_path = recipe_dir / "config.yaml"
    with config_path.open("w") as f:
        f.write(f"# Auto-generated recipe for dataset: {dataset}\n\n")
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return recipe_dir


def load_recipe(recipe_dir: Path) -> dict[str, Any]:
    """Load ``config.yaml`` from a recipe directory.

    Raises :class:`FileNotFoundError` if the config file does not exist.
    """
    config_path = recipe_dir / "config.yaml"
    with config_path.open() as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        msg = f"Expected a YAML mapping in {config_path}, got {type(result).__name__}"
        raise TypeError(msg)
    RecipeConfig.model_validate(result)
    return result


def load_recipe_config(recipe_dir: Path) -> RecipeConfig:
    """Load ``config.yaml`` from a recipe directory as a typed model."""
    return RecipeConfig.model_validate(load_recipe(recipe_dir))
