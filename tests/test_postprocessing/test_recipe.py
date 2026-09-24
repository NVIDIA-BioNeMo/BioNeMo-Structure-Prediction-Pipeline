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

"""Tests for bspp.orchestration.runtime.postprocessing.recipe."""

from __future__ import annotations

from pathlib import Path

import yaml

from bspp.orchestration.runtime.postprocessing.recipe import (
    RecipeConfig,
    generate_recipe,
    load_recipe,
    load_recipe_config,
)


def test_generate_recipe_creates_directory(tmp_path: Path) -> None:
    recipe_dir = generate_recipe("my_dataset", tmp_path)
    assert recipe_dir == tmp_path / "my_dataset"
    assert recipe_dir.is_dir()
    assert (recipe_dir / "config.yaml").is_file()


def test_generate_recipe_sets_dataset_fields(tmp_path: Path) -> None:
    recipe_dir = generate_recipe("test_10M", tmp_path)
    config = load_recipe(recipe_dir)
    assert config["run_name"] == "test_10M"
    assert config["job_name"] == "bspp_test_10M"


def test_generate_recipe_with_overrides(tmp_path: Path) -> None:
    recipe_dir = generate_recipe(
        "override_test",
        tmp_path,
        cluster="example-cluster",
        slurm={"array_range": "0-99", "time": "02:00:00"},
    )
    config = load_recipe(recipe_dir)
    assert config["cluster"] == "example-cluster"
    assert config["slurm"]["array_range"] == "0-99"
    assert config["slurm"]["time"] == "02:00:00"
    # Other slurm defaults should be preserved via deep merge
    assert "account" in config["slurm"]


def test_generate_recipe_from_template(tmp_path: Path) -> None:
    template_path = tmp_path / "template.yaml"
    template_data = {
        "cluster": "custom",
        "custom_key": "custom_value",
        "slurm": {"partition": "gpu", "time": "01:00:00"},
    }
    with template_path.open("w") as f:
        yaml.dump(template_data, f)

    recipe_dir = generate_recipe("from_template", tmp_path, template_path=template_path)
    config = load_recipe(recipe_dir)
    assert config["cluster"] == "custom"
    assert config["custom_key"] == "custom_value"
    assert config["run_name"] == "from_template"
    assert config["job_name"] == "bspp_from_template"


def test_load_recipe_roundtrip(tmp_path: Path) -> None:
    recipe_dir = generate_recipe("roundtrip", tmp_path)
    config = load_recipe(recipe_dir)
    assert isinstance(config, dict)
    assert "run_name" in config


def test_load_recipe_config_model_roundtrip(tmp_path: Path) -> None:
    recipe_dir = generate_recipe(
        "typed",
        tmp_path,
        slurm={"array_range": "0-3", "cpus_per_task": 16},
    )
    config = load_recipe_config(recipe_dir)
    assert isinstance(config, RecipeConfig)
    assert config.run_name == "typed"
    assert config.slurm is not None
    assert config.slurm.cpus_per_task == 16
    assert config.to_compat_dict()["slurm"]["array_range"] == "0-3"


def test_load_recipe_preserves_unknown_raw_mapping(tmp_path: Path) -> None:
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    (recipe_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run_name": "typed",
                "slurm": {"array_range": "0-0"},
                "custom_section": {"enabled": True},
            },
            sort_keys=False,
        )
    )

    assert load_recipe(recipe_dir) == {
        "run_name": "typed",
        "slurm": {"array_range": "0-0"},
        "custom_section": {"enabled": True},
    }


def test_load_recipe_missing_raises(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_recipe(tmp_path / "nonexistent")
