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

"""Tests for preprocess and recipe CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from bspp.orchestration.runtime.cli import cli


def test_preprocess_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["preprocess", "--help"])
    assert result.exit_code == 0
    assert "--input-dir" in result.output
    assert "--staging-dir" in result.output
    assert "--output-dir" in result.output
    assert "--manifest-csv" in result.output
    assert "--shards-per-archive" in result.output


def test_recipe_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["recipe", "--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.output
    assert "--output-dir" in result.output
    assert "--template" in result.output
