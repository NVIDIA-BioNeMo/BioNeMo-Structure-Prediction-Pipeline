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

"""Pure command builders for the legacy production pipeline adapter."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from bspp.orchestration.runtime.worker.types import PipelineCommand, PipelineResult

DsspAlgorithm = Literal["psea", "pydssp", "tmalign"]

DEFAULT_WORKERS = 24
DEFAULT_CLASH_DEVICE = "cuda"
DEFAULT_ANALYSIS_BATCH_SIZE = 128
DEFAULT_DSSP_ALGORITHM: DsspAlgorithm = "pydssp"
DEFAULT_TOOL_USED = "ColabFold v1.6.0 / AlphaFold-Multimer"


@dataclass(frozen=True, slots=True)
class ProductionPipelinePaths:
    """Filesystem inputs needed by one production_pipeline.py invocation."""

    pipeline_script: Path
    input_dir: Path
    output_dir: Path
    chain_mapping: Path
    uniprot_db: Path
    mapping_file: Path | None = None
    dataset_config: Path | None = None
    provider_json: Path | None = None


@dataclass(frozen=True, slots=True)
class ProductionPipelineOptions:
    """Legacy production_pipeline.py command options.

    The default values mirror the accepted production-pipeline invocation. The
    ``python_cmd`` field is intentionally used both as the launcher argv[0] and
    as ``--python-cmd``.
    Environment-derived defaults, such as ``CLASH_DEVICE``, must be resolved by
    CLI or RunSpec glue before constructing this typed library object.
    """

    python_cmd: str = field(default_factory=lambda: sys.executable)
    workers: int = DEFAULT_WORKERS
    clash_device: str = DEFAULT_CLASH_DEVICE
    analysis_batch_size: int = DEFAULT_ANALYSIS_BATCH_SIZE
    dssp_algorithm: DsspAlgorithm = DEFAULT_DSSP_ALGORITHM
    tool_used: str = DEFAULT_TOOL_USED
    homodimer_tool_used: str = DEFAULT_TOOL_USED
    heterodimers: bool = False
    no_heterodimers: bool = False
    parallel_stages: bool = False
    no_cache: bool = False


class PipelineCommandRunner(Protocol):
    """Dependency-injected command runner used by tests and future composition."""

    def __call__(self, command: PipelineCommand) -> PipelineResult:
        """Execute ``command`` and return a typed result."""


def production_pipeline_is_heterodimer(options: ProductionPipelineOptions) -> bool:
    """Return the effective legacy heterodimer mode."""

    return options.heterodimers and not options.no_heterodimers


def build_production_pipeline_argv(
    paths: ProductionPipelinePaths,
    options: ProductionPipelineOptions | None = None,
) -> tuple[str, ...]:
    """Build the exact ``production_pipeline.py`` argv used by the legacy worker.

    This is a pure helper. It does not check path existence, create directories,
    mutate scratch state, upload outputs, or run a subprocess.
    """

    opts = options or ProductionPipelineOptions()
    argv = [
        opts.python_cmd,
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
        opts.python_cmd,
        "--resume",
    ]

    if production_pipeline_is_heterodimer(opts):
        argv.append("--heterodimers")
    else:
        if paths.mapping_file:
            argv.extend(["--mapping-file", str(paths.mapping_file)])
        if paths.dataset_config:
            argv.extend(["--dataset-config", str(paths.dataset_config)])
        if paths.provider_json:
            argv.extend(["--provider-json", str(paths.provider_json)])

    argv.extend(["--workers", str(opts.workers)])
    argv.extend(["--clash-device", opts.clash_device])
    argv.extend(["--analysis-batch-size", str(opts.analysis_batch_size)])
    argv.extend(["--dssp-algorithm", opts.dssp_algorithm])
    argv.extend(["--tool-used", opts.tool_used])
    # The pinned public nvidia-postproc toolkit supports --homodimer-tool-used.
    # Omit its default to retain compatibility with older public toolkit builds
    # that lack this option.
    #
    # production_pipeline.py defaults this flag to DEFAULT_TOOL_USED,
    # so passing that same value is a no-op and can be dropped. Compare against
    # the default rather than against opts.tool_used: a run that sets both to a
    # non-default tool still needs the flag, or the toolkit would silently fall
    # back to its own default for homodimers.
    if opts.homodimer_tool_used != DEFAULT_TOOL_USED:
        argv.extend(["--homodimer-tool-used", opts.homodimer_tool_used])
    if opts.parallel_stages:
        argv.append("--parallel-stages")
    if opts.no_cache:
        argv.append("--no-cache")

    return tuple(argv)


def build_production_pipeline_command(
    paths: ProductionPipelinePaths,
    options: ProductionPipelineOptions | None = None,
) -> PipelineCommand:
    """Return a typed external command without executing it."""

    return PipelineCommand(argv=build_production_pipeline_argv(paths, options))


def run_production_pipeline_command(
    command: PipelineCommand,
    runner: PipelineCommandRunner,
) -> PipelineResult:
    """Run a built pipeline command through an injected runner."""

    return runner(command)


__all__ = [
    "DEFAULT_ANALYSIS_BATCH_SIZE",
    "DEFAULT_CLASH_DEVICE",
    "DEFAULT_DSSP_ALGORITHM",
    "DEFAULT_TOOL_USED",
    "DEFAULT_WORKERS",
    "DsspAlgorithm",
    "PipelineCommandRunner",
    "ProductionPipelineOptions",
    "ProductionPipelinePaths",
    "build_production_pipeline_argv",
    "build_production_pipeline_command",
    "production_pipeline_is_heterodimer",
    "run_production_pipeline_command",
]
