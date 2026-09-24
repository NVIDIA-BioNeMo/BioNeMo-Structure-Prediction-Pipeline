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

from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from bspp.orchestration.runtime.worker.pipeline_adapter import (
    DEFAULT_TOOL_USED,
    ProductionPipelineOptions,
    ProductionPipelinePaths,
    build_production_pipeline_argv,
    build_production_pipeline_command,
    production_pipeline_is_heterodimer,
    run_production_pipeline_command,
)
from bspp.orchestration.runtime.worker.types import PipelineCommand, PipelineResult


def _paths(tmp_path: Path) -> ProductionPipelinePaths:
    return ProductionPipelinePaths(
        pipeline_script=tmp_path / "AFDB-Integration-Kit" / "scripts" / "production_pipeline.py",
        input_dir=tmp_path / "work_input",
        output_dir=tmp_path / "work",
        chain_mapping=tmp_path / "work" / "shard_manifest.csv",
        uniprot_db=tmp_path / "uniprot.duckdb",
        mapping_file=tmp_path / "work" / "shard_mapping.tsv",
        dataset_config=tmp_path / "config" / "dataset_config.json",
        provider_json=tmp_path / "config" / "provider.json",
    )


def test_homodimer_tool_flag_is_omitted_when_it_matches_the_toolkit_default(tmp_path: Path) -> None:
    """Omit the no-op default for compatibility with older public toolkit builds.

    The pinned public nvidia-postproc branch supports this flag; older public
    main builds lack it and reject it during argument parsing.
    """
    paths = _paths(tmp_path)
    options = ProductionPipelineOptions(
        python_cmd="/opt/venv/bin/python",
        workers=12,
        clash_device="cpu",
        analysis_batch_size=64,
        dssp_algorithm="psea",
        tool_used="FixtureTool 1.0",
        homodimer_tool_used=DEFAULT_TOOL_USED,
    )
    assert "--homodimer-tool-used" not in build_production_pipeline_argv(paths, options)


def test_homodimer_tool_flag_is_emitted_when_it_differs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    options = ProductionPipelineOptions(
        python_cmd="/opt/venv/bin/python",
        workers=12,
        clash_device="cpu",
        analysis_batch_size=64,
        dssp_algorithm="psea",
        tool_used="FixtureTool 1.0",
        homodimer_tool_used="OtherTool 2.0",
    )
    argv = build_production_pipeline_argv(paths, options)
    assert "--homodimer-tool-used" in argv
    assert argv[argv.index("--homodimer-tool-used") + 1] == "OtherTool 2.0"


def test_build_production_pipeline_argv_matches_legacy_homodimer_order(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    options = ProductionPipelineOptions(
        python_cmd="/opt/venv/bin/python",
        workers=12,
        clash_device="cpu",
        analysis_batch_size=64,
        dssp_algorithm="psea",
        tool_used="FixtureTool 1.0",
    )

    assert build_production_pipeline_argv(paths, options) == (
        "/opt/venv/bin/python",
        str(paths.pipeline_script),
        "--input-dir",
        str(paths.input_dir),
        "--output-dir",
        str(paths.output_dir),
        "--chain-mapping",
        str(paths.chain_mapping),
        "--uniprot-db",
        str(paths.uniprot_db),
        "--python-cmd",
        "/opt/venv/bin/python",
        "--resume",
        "--mapping-file",
        str(paths.mapping_file),
        "--dataset-config",
        str(paths.dataset_config),
        "--provider-json",
        str(paths.provider_json),
        "--workers",
        "12",
        "--clash-device",
        "cpu",
        "--analysis-batch-size",
        "64",
        "--dssp-algorithm",
        "psea",
        "--tool-used",
        "FixtureTool 1.0",
    )


def test_build_production_pipeline_argv_matches_legacy_heterodimer_branch(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    options = ProductionPipelineOptions(
        python_cmd="/usr/bin/python3",
        heterodimers=True,
        parallel_stages=True,
    )

    argv = build_production_pipeline_argv(paths, options)

    assert argv == (
        "/usr/bin/python3",
        str(paths.pipeline_script),
        "--input-dir",
        str(paths.input_dir),
        "--output-dir",
        str(paths.output_dir),
        "--chain-mapping",
        str(paths.chain_mapping),
        "--uniprot-db",
        str(paths.uniprot_db),
        "--python-cmd",
        "/usr/bin/python3",
        "--resume",
        "--heterodimers",
        "--workers",
        "24",
        "--clash-device",
        "cuda",
        "--analysis-batch-size",
        "128",
        "--dssp-algorithm",
        "pydssp",
        "--tool-used",
        "ColabFold v1.6.0 / AlphaFold-Multimer",
        "--parallel-stages",
    )
    assert "--mapping-file" not in argv
    assert "--dataset-config" not in argv
    assert "--provider-json" not in argv


def test_no_heterodimers_flag_preserves_legacy_override_semantics(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    options = ProductionPipelineOptions(
        python_cmd="/usr/bin/python3",
        heterodimers=True,
        no_heterodimers=True,
    )

    argv = build_production_pipeline_argv(paths, options)

    assert production_pipeline_is_heterodimer(options) is False
    assert "--heterodimers" not in argv
    assert "--mapping-file" in argv


def test_no_cache_option_disables_legacy_pipeline_cache(tmp_path: Path) -> None:
    argv = build_production_pipeline_argv(
        _paths(tmp_path),
        ProductionPipelineOptions(python_cmd="/usr/bin/python3", no_cache=True),
    )

    assert argv[-1] == "--no-cache"


def test_clash_device_is_typed_input_not_read_from_environment(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("CLASH_DEVICE", "mps")

    argv = build_production_pipeline_argv(_paths(tmp_path), ProductionPipelineOptions(python_cmd="/usr/bin/python3"))

    assert argv[argv.index("--clash-device") + 1] == "cuda"


def test_build_production_pipeline_command_is_typed_and_pure(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    options = ProductionPipelineOptions(python_cmd="/usr/bin/python3")
    command = build_production_pipeline_command(paths, options)

    assert command == PipelineCommand(argv=build_production_pipeline_argv(paths, options))
    assert command.working_dir is None
    assert command.env == ()
    assert not paths.output_dir.exists()
    assert options.python_cmd == "/usr/bin/python3"


def test_run_production_pipeline_command_uses_injected_runner(tmp_path: Path) -> None:
    command = build_production_pipeline_command(
        _paths(tmp_path),
        ProductionPipelineOptions(python_cmd="/usr/bin/python3"),
    )
    seen_commands: list[PipelineCommand] = []

    def fake_runner(command: PipelineCommand) -> PipelineResult:
        seen_commands.append(command)
        return PipelineResult(exit_code=7, elapsed_seconds=0.25)

    result = run_production_pipeline_command(command, fake_runner)

    assert seen_commands == [command]
    assert result == PipelineResult(exit_code=7, elapsed_seconds=0.25)
