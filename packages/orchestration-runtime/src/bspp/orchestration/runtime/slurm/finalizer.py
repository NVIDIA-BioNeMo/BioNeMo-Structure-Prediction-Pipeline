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

"""Render SLURM analysis metadata finalizer scripts from RunSpec."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.runspec import MountSpec, RunSpec, render_artifact_header
from bspp.orchestration.contract.runspec_validation import ORCHESTRATION_CONTAINER_TARGET, has_mount_target
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.slurm.native_contract import NativeRuntimeContract, build_native_runtime_contract


@dataclass(frozen=True)
class AnalysisFinalizerPlan:
    """Rendered analysis finalizer script and operator command."""

    script_path: Path
    log_dir: Path
    command: tuple[str, ...]
    dependency_command: tuple[str, ...] | None
    dry_run: bool

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable plan data."""
        return {
            "script_path": str(self.script_path),
            "log_dir": str(self.log_dir),
            "command": list(self.command),
            "dependency_command": list(self.dependency_command) if self.dependency_command else None,
            "dry_run": self.dry_run,
        }


def render_analysis_finalizer(
    spec: RunSpec,
    *,
    dry_run: bool = True,
    script_path: Path | None = None,
    dependency_job_id: str | None = None,
) -> AnalysisFinalizerPlan:
    """Render or plan a containerized analysis metadata finalizer script."""
    _validate_finalizer_spec(spec)
    script = script_path or spec.paths.output_dir / "wp5" / "run_analysis_finalize.sbatch"
    command = ("sbatch", str(script))
    dependency_command = (
        ("sbatch", f"--dependency=afterok:{dependency_job_id}", str(script)) if dependency_job_id else None
    )
    plan = AnalysisFinalizerPlan(
        script_path=script,
        log_dir=spec.paths.log_dir,
        command=command,
        dependency_command=dependency_command,
        dry_run=dry_run,
    )
    if not dry_run:
        spec.paths.log_dir.mkdir(parents=True, exist_ok=True)
        _write_text(_render_script(spec), script)
        report = {
            "run_id": spec.dataset.run_id,
            "dataset": spec.dataset.name,
            "source_runspec": str(spec.source_path) if spec.source_path is not None else None,
            "source_hash": spec.source_hash,
            "plan": plan,
        }
        write_json_report(report, script.parent / "analysis_finalizer_report.json")
        write_text_summary(report, script.parent / "analysis_finalizer_report.txt")
    return plan


def render_analysis_finalizer_report(spec: RunSpec, plan: AnalysisFinalizerPlan) -> str:
    """Render a deterministic JSON report for an analysis finalizer plan."""
    return report_to_json(
        {
            "run_id": spec.dataset.run_id,
            "dataset": spec.dataset.name,
            "source_runspec": str(spec.source_path) if spec.source_path is not None else None,
            "source_hash": spec.source_hash,
            "plan": plan,
        }
    )


def _validate_finalizer_spec(spec: RunSpec) -> None:
    metadata = spec.analysis_metadata
    if not metadata.enabled:
        msg = "analysis_metadata.enabled must be true to render the finalizer"
        raise ValueError(msg)
    if not metadata.csv_path or not metadata.parquet_path or not metadata.selected_ids_path:
        msg = "analysis metadata finalizer requires csv_path, parquet_path, and selected_ids_path"
        raise ValueError(msg)
    hq = metadata.high_quality_from_tars
    if hq.enabled and (not hq.s3_prefix or not hq.work_dir):
        msg = "high_quality_from_tars requires s3_prefix and work_dir when enabled"
        raise ValueError(msg)


def _render_script(spec: RunSpec) -> str:
    metadata = spec.analysis_metadata
    resource = spec.resources.get("analysis_finalize")
    partition = resource.partition if resource else metadata.finalize_partition or "cpu_long"
    cpus = resource.cpus_per_task if resource else metadata.finalize_cpus_per_task
    memory = resource.memory if resource else metadata.finalize_memory
    time = resource.time if resource else metadata.finalize_time
    lines = [
        "#!/bin/bash",
        render_artifact_header(spec, "slurm/run_analysis_finalize.sbatch").rstrip(),
        "#SBATCH --job-name=" + _shell_word(f"bspp_{spec.dataset.name}_analysis_finalize"),
        f"#SBATCH --partition={partition}",
        f"#SBATCH --account={spec.cluster.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={spec.paths.log_dir / 'analysis_metadata_finalize_%j.out'}",
        f"#SBATCH --error={spec.paths.log_dir / 'analysis_metadata_finalize_%j.err'}",
        "",
        "set -euo pipefail",
        "",
        'echo "BSPP analysis finalizer: ${SLURM_JOB_ID:-local}"',
        _srun_command(spec),
        "",
    ]
    return "\n".join(lines)


def _srun_command(spec: RunSpec) -> str:
    runtime = build_native_runtime_contract(spec)
    dev_env = (
        ["env", "BSPP_ORCHESTRATION_DEV_MOUNT=1"] if has_mount_target(spec, ORCHESTRATION_CONTAINER_TARGET) else []
    )
    srun_argv = [
        "srun",
        f"--container-image={spec.container.image}",
        f"--container-mounts={_container_mounts(spec)}",
        "--no-container-mount-home",
        *dev_env,
        "/usr/local/bin/entrypoint.sh",
        "bash",
    ]
    container_script = _container_script(
        spec,
        runtime=runtime,
    )
    return (
        " \\\n  ".join(shlex.quote(part) for part in srun_argv)
        + " <<'BSPP_FINALIZE_IN_CONTAINER'\n"
        + container_script
        + "\nBSPP_FINALIZE_IN_CONTAINER"
    )


def _container_script(
    spec: RunSpec,
    *,
    runtime: NativeRuntimeContract,
) -> str:
    metadata = spec.analysis_metadata
    assert metadata.csv_path is not None
    assert metadata.parquet_path is not None
    assert metadata.selected_ids_path is not None
    lines = [
        "set -euo pipefail",
        "",
        f"ORCHESTRATION_ROOT={shlex.quote(str(runtime.orchestration_container_path))}",
        f"CONTAINER_WORKDIR={shlex.quote(str(runtime.container_workdir))}",
        f"S5CMD_PATH={shlex.quote(str(spec.worker.s5cmd_path))}",
        f'PYTHON_BIN="${{BSPP_ORCHESTRATION_PYTHON:-{runtime.pixi_python_path}}}"',
        'fail() { echo "BSPP finalizer preflight failed: $*" >&2; exit 127; }',
        'require_dir() { [[ -d "$1" ]] || fail "missing $2: $1"; }',
        'require_file() { [[ -f "$1" ]] || fail "missing $2: $1"; }',
        'require_executable() { [[ -x "$1" ]] || fail "missing executable $2: $1"; }',
        "",
        'require_dir "$ORCHESTRATION_ROOT" "orchestration mount"',
        'require_dir "$CONTAINER_WORKDIR" "container workdir"',
        'if [[ ! -x "$PYTHON_BIN" ]]; then',
        '  PYTHON_BIN="$(command -v python || true)"',
        "fi",
        'require_executable "$PYTHON_BIN" "Python runtime"',
        (
            'export PYTHONPATH="${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:'
            '${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src${PYTHONPATH:+:${PYTHONPATH}}"'
        ),
        '"$PYTHON_BIN" -c "import bspp.orchestration.runtime.postprocessing.analysis_finalizer" >/dev/null',
        'cd "$CONTAINER_WORKDIR"',
        "",
    ]
    if metadata.high_quality_from_tars.enabled:
        lines.extend(
            [
                'if [[ "$S5CMD_PATH" == */* ]]; then',
                '  require_executable "$S5CMD_PATH" "configured s5cmd path"',
                "else",
                '  command -v "$S5CMD_PATH" >/dev/null 2>&1 || fail "missing configured s5cmd on PATH: $S5CMD_PATH"',
                "fi",
                "",
            ]
        )
    lines.append(
        _finalize_command(
            spec,
            csv_path=metadata.csv_path,
            parquet_path=metadata.parquet_path,
            selected_ids_path=metadata.selected_ids_path,
        )
    )
    if metadata.high_quality_from_tars.enabled:
        lines.extend(["", _high_quality_note_command(spec)])
    return "\n".join(lines)


def _finalize_command(
    spec: RunSpec,
    *,
    csv_path: Path,
    parquet_path: Path,
    selected_ids_path: Path,
) -> str:
    parts = [
        "-m",
        "bspp.orchestration.runtime.postprocessing.analysis_finalizer",
        "--csv",
        str(csv_path),
        "--parquet",
        str(parquet_path),
        "--selected-ids",
        str(selected_ids_path),
    ]
    if spec.analysis_metadata.high_quality_from_tars.enabled:
        hq = spec.analysis_metadata.high_quality_from_tars
        assert hq.s3_prefix is not None
        assert hq.work_dir is not None
        tar_manifest = spec.storage.local_tar_manifest_csv or spec.storage.s3_tar_manifest_csv
        if tar_manifest is None:
            msg = "high_quality_from_tars requires a local or S3 tar manifest CSV"
            raise ValueError(msg)
        parts.extend(
            [
                "--local-tars-csv",
                str(tar_manifest),
                "--model-tar-index",
                str(spec.paths.output_dir / "model_tar_index.csv"),
                "--high-quality-work-dir",
                str(hq.work_dir),
                "--high-quality-s3-prefix",
                hq.s3_prefix,
                "--s5cmd-path",
                str(spec.worker.s5cmd_path),
                "--s5cmd-numworkers",
                str(spec.worker.s5cmd_numworkers),
            ]
        )
        if spec.storage.local_tar_dir is not None:
            parts.extend(["--local-tar-dir", str(spec.storage.local_tar_dir)])
    return '"$PYTHON_BIN" \\\n  ' + " \\\n  ".join(shlex.quote(part) for part in parts)


def _high_quality_note_command(spec: RunSpec) -> str:
    hq = spec.analysis_metadata.high_quality_from_tars
    assert hq.s3_prefix is not None
    return f'echo "BSPP high-quality extraction/upload completed for {shlex.quote(hq.s3_prefix)}"'


def _container_mounts(spec: RunSpec) -> str:
    mounts = [MountSpec(source=Path("/lustre"), target=Path("/lustre"), read_only=False)]
    mounts.extend(spec.container.mounts)
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


__all__ = ["AnalysisFinalizerPlan", "render_analysis_finalizer", "render_analysis_finalizer_report"]
