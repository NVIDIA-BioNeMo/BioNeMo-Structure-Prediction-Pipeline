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

"""Render acceptance comparator SLURM scripts from RunSpec."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.runspec import (
    AcceptanceSpec,
    RunSpec,
    SlurmResources,
    SubmissionSpec,
    render_artifact_header,
)
from bspp.orchestration.contract.runspec_validation import ORCHESTRATION_CONTAINER_TARGET, has_mount_target
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary
from bspp.orchestration.runtime.slurm.native_contract import (
    NativeRuntimeContract,
    all_container_mounts,
    build_native_runtime_contract,
)

AcceptanceCheckName = Literal["tar-payload-parity", "semantic"]


@dataclass(frozen=True)
class AcceptanceSbatchPlan:
    """Rendered acceptance comparator script and report locations."""

    step_name: str
    check_name: AcceptanceCheckName
    script_path: Path
    evidence_dir: Path
    report_path: Path
    text_report_path: Path
    log_patterns: tuple[str, str]
    command: tuple[str, ...]
    validation_command: tuple[str, ...]
    dry_run: bool

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-serializable plan data."""
        return {
            "step_name": self.step_name,
            "check_name": self.check_name,
            "script_path": str(self.script_path),
            "evidence_dir": str(self.evidence_dir),
            "report_path": str(self.report_path),
            "text_report_path": str(self.text_report_path),
            "log_patterns": list(self.log_patterns),
            "command": list(self.command),
            "validation_command": list(self.validation_command),
            "dry_run": self.dry_run,
        }


def render_acceptance_tar_payload_parity(
    spec: RunSpec,
    *,
    dry_run: bool = True,
    script_path: Path | None = None,
) -> AcceptanceSbatchPlan:
    """Render or plan the tar-payload parity acceptance comparator script."""
    submission, acceptance = _required_submission_acceptance(spec)
    if acceptance.baseline_output_dir is None:
        msg = "acceptance.baseline_output_dir is required for tar payload parity"
        raise ValueError(msg)
    resource = _required_resource(spec, "acceptance_tar_payload_parity")
    evidence_dir = submission.evidence_dir / "acceptance" / "tar_payload_parity"
    script = script_path or evidence_dir / "run_tar_payload_parity.sbatch"
    command = _tar_payload_command(spec)
    plan = AcceptanceSbatchPlan(
        step_name="acceptance-tar-payload-parity",
        check_name="tar-payload-parity",
        script_path=script,
        evidence_dir=evidence_dir,
        report_path=evidence_dir / "tar_payload_parity_report.json",
        text_report_path=evidence_dir / "tar_payload_parity_report.txt",
        log_patterns=(str(evidence_dir / "tar_payload_parity_%j.out"), str(evidence_dir / "tar_payload_parity_%j.err")),
        command=("sbatch", "--parsable", str(script)),
        validation_command=command,
        dry_run=dry_run,
    )
    if not dry_run:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        runtime = build_native_runtime_contract(spec)
        _write_text(
            _render_script(
                spec,
                plan,
                resource=resource,
                runtime=runtime,
                artifact_name="slurm/acceptance/tar_payload_parity/run_tar_payload_parity.sbatch",
                job_suffix="acceptance_tar_payload_parity",
                require_zstd=True,
            ),
            script,
        )
        _write_plan_reports(spec, plan, "tar_payload_parity_sbatch_report")
    return plan


def render_acceptance_semantic(
    spec: RunSpec,
    *,
    dry_run: bool = True,
    script_path: Path | None = None,
) -> AcceptanceSbatchPlan:
    """Render or plan the semantic acceptance comparator script."""
    submission, acceptance = _required_submission_acceptance(spec)
    if acceptance.baseline_output_dir is None:
        msg = "acceptance.baseline_output_dir is required for semantic acceptance"
        raise ValueError(msg)
    resource = _required_resource(spec, "acceptance_semantic")
    evidence_dir = submission.evidence_dir / "acceptance" / "semantic_acceptance"
    script = script_path or evidence_dir / "run_semantic_acceptance.sbatch"
    command = _semantic_command(spec)
    plan = AcceptanceSbatchPlan(
        step_name="acceptance-semantic",
        check_name="semantic",
        script_path=script,
        evidence_dir=evidence_dir,
        report_path=evidence_dir / "semantic_acceptance_summary.json",
        text_report_path=evidence_dir / "semantic_acceptance_summary.txt",
        log_patterns=(
            str(evidence_dir / "semantic_acceptance_%j.out"),
            str(evidence_dir / "semantic_acceptance_%j.err"),
        ),
        command=("sbatch", "--parsable", str(script)),
        validation_command=command,
        dry_run=dry_run,
    )
    if not dry_run:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        runtime = build_native_runtime_contract(spec)
        _write_text(
            _render_script(
                spec,
                plan,
                resource=resource,
                runtime=runtime,
                artifact_name="slurm/acceptance/semantic_acceptance/run_semantic_acceptance.sbatch",
                job_suffix="acceptance_semantic",
                require_zstd=False,
            ),
            script,
        )
        _write_plan_reports(spec, plan, "semantic_acceptance_sbatch_report")
    return plan


def render_acceptance_sbatch_report(spec: RunSpec, plan: AcceptanceSbatchPlan) -> str:
    """Render a deterministic JSON report for an acceptance sbatch plan."""
    return report_to_json(_plan_report(spec, plan))


def _render_script(
    spec: RunSpec,
    plan: AcceptanceSbatchPlan,
    *,
    resource: SlurmResources,
    runtime: NativeRuntimeContract,
    artifact_name: str,
    job_suffix: str,
    require_zstd: bool,
) -> str:
    lines = [
        "#!/bin/bash",
        render_artifact_header(spec, artifact_name).rstrip(),
        "#SBATCH --job-name=" + _shell_word(f"bspp_{spec.dataset.name}_{job_suffix}"),
        f"#SBATCH --partition={resource.partition}",
        f"#SBATCH --account={spec.cluster.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resource.cpus_per_task}",
        f"#SBATCH --mem={resource.memory}",
        f"#SBATCH --time={resource.time}",
        f"#SBATCH --output={plan.log_patterns[0]}",
        f"#SBATCH --error={plan.log_patterns[1]}",
        "",
        "set -euo pipefail",
        "",
        f'echo "BSPP {plan.check_name} acceptance: ${{SLURM_JOB_ID:-local}}"',
        _srun_command(spec, plan, runtime=runtime, require_zstd=require_zstd),
        "",
    ]
    return "\n".join(lines)


def _srun_command(
    spec: RunSpec,
    plan: AcceptanceSbatchPlan,
    *,
    runtime: NativeRuntimeContract,
    require_zstd: bool,
) -> str:
    dev_env = (
        ["env", "BSPP_ORCHESTRATION_DEV_MOUNT=1"] if has_mount_target(spec, ORCHESTRATION_CONTAINER_TARGET) else []
    )
    srun_argv = [
        "srun",
        f"--container-image={spec.container.image}",
        f"--container-mounts={_container_mounts(spec, runtime)}",
        "--no-container-mount-home",
        *dev_env,
        "/usr/local/bin/entrypoint.sh",
        "bash",
    ]
    return (
        " \\\n  ".join(shlex.quote(part) for part in srun_argv)
        + " <<'BSPP_ACCEPTANCE_IN_CONTAINER'\n"
        + _container_script(plan, runtime=runtime, require_zstd=require_zstd)
        + "\nBSPP_ACCEPTANCE_IN_CONTAINER"
    )


def _container_script(
    plan: AcceptanceSbatchPlan,
    *,
    runtime: NativeRuntimeContract,
    require_zstd: bool,
) -> str:
    lines = [
        "set -euo pipefail",
        "",
        f"ORCHESTRATION_ROOT={shlex.quote(str(runtime.orchestration_container_path))}",
        f"CONTAINER_WORKDIR={shlex.quote(str(runtime.container_workdir))}",
        f'PYTHON_BIN="${{BSPP_ORCHESTRATION_PYTHON:-{runtime.pixi_python_path}}}"',
        'fail() { echo "BSPP acceptance preflight failed: $*" >&2; exit 127; }',
        'require_dir() { [[ -d "$1" ]] || fail "missing $2: $1"; }',
        'require_executable() { [[ -x "$1" ]] || fail "missing executable $2: $1"; }',
        "",
        'require_dir "$ORCHESTRATION_ROOT" "orchestration mount"',
        'require_dir "$CONTAINER_WORKDIR" "container workdir"',
        'if [[ ! -x "$PYTHON_BIN" ]]; then',
        '  PYTHON_BIN="$(command -v python || true)"',
        "fi",
        'require_executable "$PYTHON_BIN" "Python runtime"',
    ]
    if require_zstd:
        lines.append('command -v zstd >/dev/null 2>&1 || fail "missing zstd on PATH"')
    lines.extend(
        [
            (
                'export PYTHONPATH="${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:'
                '${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src${PYTHONPATH:+:${PYTHONPATH}}"'
            ),
            '"$PYTHON_BIN" -c "from bspp.orchestration.runtime.cli import cli; cli()" --help >/dev/null',
            'cd "$CONTAINER_WORKDIR"',
            f"mkdir -p {shlex.quote(str(plan.evidence_dir))}",
            "",
            _cli_command(plan.validation_command),
        ]
    )
    return "\n".join(lines)


def _tar_payload_command(spec: RunSpec) -> tuple[str, ...]:
    assert spec.acceptance is not None
    assert spec.acceptance.baseline_output_dir is not None
    command = [
        "validate",
        "tar-payload-parity",
        "--baseline-dir",
        str(spec.acceptance.baseline_output_dir),
        "--candidate-dir",
        str(spec.paths.output_dir),
        "--relative-dir",
        "local_tars",
        "--exclude",
        "metadata/",
        "--match-mode",
        spec.acceptance.tar_payload_match_mode,
        "--workers",
        "${SLURM_CPUS_PER_TASK}",
    ]
    if spec.acceptance.baseline_run_name is not None:
        command.extend(["--baseline-run-name", spec.acceptance.baseline_run_name])
    if spec.acceptance.candidate_run_name is not None:
        command.extend(["--candidate-run-name", spec.acceptance.candidate_run_name])
    if spec.acceptance.payload_sample_count is not None:
        command.extend(["--payload-sample-count", str(spec.acceptance.payload_sample_count)])
    report_dir = _required_submission_acceptance(spec)[0].evidence_dir / "acceptance" / "tar_payload_parity"
    command.extend(["--write-report", str(report_dir)])
    command.append("--strict")
    return tuple(command)


def _semantic_command(spec: RunSpec) -> tuple[str, ...]:
    assert spec.acceptance is not None
    assert spec.acceptance.baseline_output_dir is not None
    command = [
        "validate",
        "semantic-acceptance",
        "--baseline-dir",
        str(spec.acceptance.baseline_output_dir),
        "--candidate-dir",
        str(spec.paths.output_dir),
    ]
    optional_counts = (
        ("--expected-tar-count", spec.validation.expected_tar_count),
        ("--expected-local-tars-rows", spec.validation.expected_local_tars_rows),
        ("--expected-failed-rows", spec.validation.expected_failed_rows),
        ("--expected-analysis-rows", spec.validation.expected_analysis_rows),
        ("--expected-selected-ids", spec.validation.expected_selected_ids),
    )
    for flag, value in optional_counts:
        if value is not None:
            command.extend([flag, str(value)])
    if not spec.acceptance.candidate_parquet_required:
        command.append("--candidate-parquet-optional")
    if not spec.acceptance.compare_failed_sets:
        command.append("--no-compare-failed-sets")
    if not spec.acceptance.compare_tar_manifest_rows:
        command.append("--no-compare-tar-manifest-rows")
    if not spec.acceptance.compare_analysis_model_rows:
        command.append("--no-compare-analysis-model-rows")
    report_dir = _required_submission_acceptance(spec)[0].evidence_dir / "acceptance" / "semantic_acceptance"
    command.extend(["--write-report", str(report_dir)])
    command.append("--strict")
    return tuple(command)


def _cli_command(validation_command: tuple[str, ...]) -> str:
    parts = ['"$PYTHON_BIN"', "-c", shlex.quote("from bspp.orchestration.runtime.cli import cli; cli()")]
    parts.extend(_shell_arg(part) for part in validation_command)
    return " \\\n  ".join(parts)


def _shell_arg(part: str) -> str:
    if part == "${SLURM_CPUS_PER_TASK}":
        return '"${SLURM_CPUS_PER_TASK}"'
    return shlex.quote(part)


def _required_submission_acceptance(spec: RunSpec) -> tuple[SubmissionSpec, AcceptanceSpec]:
    if spec.submission is None:
        msg = "submission is required to render acceptance scripts"
        raise ValueError(msg)
    if spec.acceptance is None:
        msg = "acceptance is required to render acceptance scripts"
        raise ValueError(msg)
    return spec.submission, spec.acceptance


def _required_resource(spec: RunSpec, key: str) -> SlurmResources:
    resource = spec.resources.get(key)
    if resource is None:
        msg = f"resources.{key} is required to render acceptance scripts"
        raise ValueError(msg)
    return resource


def _container_mounts(spec: RunSpec, runtime: NativeRuntimeContract) -> str:
    mounts = all_container_mounts(spec, runtime.runspec_container_path)
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


def _write_plan_reports(spec: RunSpec, plan: AcceptanceSbatchPlan, stem: str) -> None:
    report = _plan_report(spec, plan)
    write_json_report(report, plan.script_path.parent / f"{stem}.json")
    write_text_summary(report, plan.script_path.parent / f"{stem}.txt")


def _plan_report(spec: RunSpec, plan: AcceptanceSbatchPlan) -> dict[str, object]:
    return {
        "run_id": spec.dataset.run_id,
        "dataset": spec.dataset.name,
        "source_runspec": str(spec.source_path) if spec.source_path is not None else None,
        "source_hash": spec.source_hash,
        "plan": plan,
    }


def _write_text(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _shell_word(value: str) -> str:
    return value.replace("/", "_").replace("-", "_")


__all__ = [
    "AcceptanceSbatchPlan",
    "render_acceptance_sbatch_report",
    "render_acceptance_semantic",
    "render_acceptance_tar_payload_parity",
]
