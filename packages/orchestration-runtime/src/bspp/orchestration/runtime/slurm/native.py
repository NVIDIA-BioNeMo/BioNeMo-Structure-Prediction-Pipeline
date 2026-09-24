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

"""Render containerized native archive-worker SLURM scripts from RunSpec."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.runspec import RunSpec, render_artifact_header
from bspp.orchestration.contract.runspec_validation import ORCHESTRATION_CONTAINER_TARGET, has_mount_target
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.postprocessing.runspec_artifacts import normalized_archive_names
from bspp.orchestration.runtime.slurm.arrays import effective_archive_array
from bspp.orchestration.runtime.slurm.native_contract import (
    NativeRuntimeContract,
    all_container_mounts,
    build_native_runtime_contract,
)


@dataclass(frozen=True)
class SlurmNativePlan:
    """Rendered native SLURM worker script and operator commands."""

    script_path: Path
    log_dir: Path
    array_range: str
    runspec_container_path: Path
    toolkit_container_path: Path
    orchestration_container_path: Path
    pixi_python_path: Path
    pythonpath_entries: tuple[Path, ...]
    required_tools: tuple[str, ...]
    production_pipeline_container_path: Path
    one_archive_command: tuple[str, ...]
    full_array_command: tuple[str, ...]
    monitor_command: tuple[str, ...]
    dry_run: bool

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable plan data."""
        return {
            "script_path": str(self.script_path),
            "log_dir": str(self.log_dir),
            "array_range": self.array_range,
            "runspec_container_path": str(self.runspec_container_path),
            "toolkit_container_path": str(self.toolkit_container_path),
            "orchestration_container_path": str(self.orchestration_container_path),
            "pixi_python_path": str(self.pixi_python_path),
            "pythonpath_entries": [str(path) for path in self.pythonpath_entries],
            "required_tools": list(self.required_tools),
            "production_pipeline_container_path": str(self.production_pipeline_container_path),
            "one_archive_command": list(self.one_archive_command),
            "full_array_command": list(self.full_array_command),
            "monitor_command": list(self.monitor_command),
            "dry_run": self.dry_run,
        }


def render_slurm_native(
    spec: RunSpec,
    archives: list[str] | tuple[str, ...],
    *,
    dry_run: bool = True,
    script_path: Path | None = None,
) -> SlurmNativePlan:
    """Render or plan a containerized native archive-worker SLURM script."""
    archive_names = normalized_archive_names(archives)
    if not archive_names:
        msg = "At least one archive is required to render the native SLURM worker"
        raise ValueError(msg)
    runtime = build_native_runtime_contract(spec)
    array_range = effective_archive_array(spec, len(archive_names))
    script = script_path or spec.paths.output_dir / "wp5" / "run_archive.sbatch"
    one_archive_command = ("sbatch", "--array=0-0", str(script))
    full_array_command = ("sbatch", f"--array={array_range}", str(script))
    monitor_command = ("squeue", "-u", spec.cluster.owner or "$USER", "-n", f"bspp_{spec.dataset.name}_native")
    plan = SlurmNativePlan(
        script_path=script,
        log_dir=spec.paths.log_dir,
        array_range=array_range,
        runspec_container_path=runtime.runspec_container_path,
        toolkit_container_path=runtime.toolkit_container_path,
        orchestration_container_path=runtime.orchestration_container_path,
        pixi_python_path=runtime.pixi_python_path,
        pythonpath_entries=runtime.pythonpath_entries,
        required_tools=runtime.required_tools,
        production_pipeline_container_path=runtime.production_pipeline_container_path,
        one_archive_command=one_archive_command,
        full_array_command=full_array_command,
        monitor_command=monitor_command,
        dry_run=dry_run,
    )
    if not dry_run:
        spec.paths.log_dir.mkdir(parents=True, exist_ok=True)
        _write_text(_render_script(spec, archive_names, array_range, runtime), script)
        report = _native_report(spec, plan)
        write_json_report(report, script.parent / "slurm_native_report.json")
        write_text_summary(report, script.parent / "slurm_native_report.txt")
    return plan


def render_slurm_native_report(spec: RunSpec, plan: SlurmNativePlan) -> str:
    """Render a deterministic JSON report for a native SLURM plan."""
    return report_to_json(_native_report(spec, plan))


def _native_report(spec: RunSpec, plan: SlurmNativePlan) -> dict[str, object]:
    return {
        "run_id": spec.dataset.run_id,
        "dataset": spec.dataset.name,
        "source_runspec": str(spec.source_path) if spec.source_path is not None else None,
        "source_hash": spec.source_hash,
        "plan": plan,
    }


def _render_script(
    spec: RunSpec,
    archive_names: tuple[str, ...],
    array_range: str,
    runtime: NativeRuntimeContract,
) -> str:
    _ = archive_names
    gpu = spec.resources["gpu_worker"]
    lines = [
        "#!/bin/bash",
        render_artifact_header(spec, "slurm/run_archive.sbatch").rstrip(),
        "#SBATCH --job-name=" + _shell_word(f"bspp_{spec.dataset.name}_native"),
        f"#SBATCH --partition={gpu.partition}",
        f"#SBATCH --account={spec.cluster.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={gpu.cpus_per_task}",
        f"#SBATCH --mem={gpu.memory}",
        f"#SBATCH --time={gpu.time}",
        f"#SBATCH --array={array_range}",
        f"#SBATCH --output={spec.paths.log_dir / 'native_%A_%a.out'}",
        f"#SBATCH --error={spec.paths.log_dir / 'native_%A_%a.err'}",
    ]
    if gpu.gres:
        lines.append(f"#SBATCH --gres={gpu.gres}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f"export DUCKDB_MEMORY_LIMIT={shlex.quote(spec.worker.duckdb_memory_limit)}",
            "",
            'echo "BSPP native archive worker: ${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"',
            _srun_command(spec, runtime),
            "",
        ]
    )
    return "\n".join(lines)


def _srun_command(spec: RunSpec, runtime: NativeRuntimeContract) -> str:
    dev_env = (
        ["env", "BSPP_ORCHESTRATION_DEV_MOUNT=1"] if has_mount_target(spec, ORCHESTRATION_CONTAINER_TARGET) else []
    )
    srun_argv = [
        "srun",
        f"--container-image={spec.container.image}",
        f"--container-mounts={_container_mounts(spec, runtime.runspec_container_path)}",
        "--no-container-mount-home",
        *dev_env,
        "/usr/local/bin/entrypoint.sh",
        "bash",
    ]
    container_script = _container_script(
        runtime=runtime,
        s5cmd_path=spec.worker.s5cmd_path,
    )
    return (
        " \\\n  ".join(shlex.quote(part) for part in srun_argv)
        + " <<'BSPP_NATIVE_IN_CONTAINER'\n"
        + container_script
        + "\nBSPP_NATIVE_IN_CONTAINER"
    )


def _container_script(
    *,
    runtime: NativeRuntimeContract,
    s5cmd_path: str,
) -> str:
    lines = [
        "set -euo pipefail",
        "",
        f"TOOLKIT_ROOT={shlex.quote(str(runtime.toolkit_container_path))}",
        f"ORCHESTRATION_ROOT={shlex.quote(str(runtime.orchestration_container_path))}",
        f"CONTAINER_WORKDIR={shlex.quote(str(runtime.container_workdir))}",
        f"RUNSPEC={shlex.quote(str(runtime.runspec_container_path))}",
        f"PRODUCTION_PIPELINE={shlex.quote(str(runtime.production_pipeline_container_path))}",
        f"S5CMD_PATH={shlex.quote(str(s5cmd_path))}",
        f'PYTHON_BIN="${{BSPP_ORCHESTRATION_PYTHON:-{runtime.pixi_python_path}}}"',
        'fail() { echo "BSPP native worker preflight failed: $*" >&2; exit 127; }',
        'require_dir() { [[ -d "$1" ]] || fail "missing $2: $1"; }',
        'require_file() { [[ -f "$1" ]] || fail "missing $2: $1"; }',
        'require_executable() { [[ -x "$1" ]] || fail "missing executable $2: $1"; }',
        "",
        'require_dir "$TOOLKIT_ROOT" "toolkit mount"',
        'require_dir "$ORCHESTRATION_ROOT" "orchestration mount"',
        'require_dir "$CONTAINER_WORKDIR" "container workdir"',
        'require_file "$RUNSPEC" "RunSpec"',
        'require_file "$PRODUCTION_PIPELINE" "production_pipeline.py"',
        'if [[ ! -x "$PYTHON_BIN" ]]; then',
        '  PYTHON_BIN="$(command -v python || true)"',
        "fi",
        'require_executable "$PYTHON_BIN" "Python runtime"',
        'command -v tar >/dev/null 2>&1 || fail "missing tar on PATH"',
        'command -v lz4 >/dev/null 2>&1 || fail "missing lz4 on PATH"',
    ]
    if runtime.zstd_required:
        lines.append('command -v zstd >/dev/null 2>&1 || fail "missing zstd on PATH"')
    if runtime.s5cmd_required:
        lines.extend(
            [
                'if [[ "$S5CMD_PATH" == */* ]]; then',
                '  require_executable "$S5CMD_PATH" "configured s5cmd path"',
                "else",
                '  command -v "$S5CMD_PATH" >/dev/null 2>&1 || fail "missing configured s5cmd on PATH: $S5CMD_PATH"',
                "fi",
            ]
        )
    lines.extend(
        [
            (
                'export PYTHONPATH="${TOOLKIT_ROOT}:${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:'
                '${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src${PYTHONPATH:+:${PYTHONPATH}}"'
            ),
            '"$PYTHON_BIN" -c "from bspp.orchestration.runtime.cli import cli; cli()" --help >/dev/null',
            'cd "$CONTAINER_WORKDIR"',
            "",
            _native_worker_command(),
        ]
    )
    return "\n".join(lines)


def _native_worker_command() -> str:
    return " \\\n  ".join(
        (
            '"$PYTHON_BIN"',
            "-c",
            shlex.quote("from bspp.orchestration.runtime.cli import cli; cli()"),
            "worker",
            "archive-task",
            "--runspec",
            '"$RUNSPEC"',
        )
    )


def _container_mounts(spec: RunSpec, runspec_container_path: Path) -> str:
    mounts = all_container_mounts(spec, runspec_container_path)
    seen: set[tuple[str, str]] = set()
    rendered: list[str] = []
    for mount in mounts:
        key = (str(mount.source), str(mount.target))
        if key in seen:
            continue
        seen.add(key)
        suffix = ":ro" if mount.read_only else ""
        rendered.append(f"{mount.source}:{mount.target}{suffix}")
    return ",".join(rendered)


def _write_text(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _shell_word(value: str) -> str:
    return value.replace("/", "_")
