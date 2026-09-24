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

"""Control Plane rendering for containerized workflow Slurm scripts."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.runspec import DataMoverSelectionSpec, RunSpec, SlurmResources, WorkflowStepSpec
from bspp.orchestration.contract.runspec_policies import provenance_governed
from bspp.orchestration.contract.runspec_validation import ORCHESTRATION_CONTAINER_TARGET, has_mount_target
from bspp.orchestration.contract.source_package import SourcePackageIdentity
from bspp.orchestration.control.execution_bootstrap import ImageIdentity, render_governed_srun
from bspp.orchestration.control.execution_provenance_events import render_runtime_result_epilogue
from bspp.orchestration.control.governed_submission import stable_read_descendant, write_immutable_descendant

RUNSPEC_CONTAINER_PATH = Path("/workspace/bspp-runspec/runspec.yaml")
LEGACY_RUNSPEC_CONTAINER_PATH = Path("/workspace/afcdb-runspec/runspec.yaml")
ORCHESTRATION_CONTAINER_ROOT = Path("/workspace/bspp-orchestration")
TOOLKIT_CONTAINER_ROOT = Path("/workspace/AFDB-Integration-Kit")
BAKED_TOOLKIT_CONTAINER_ROOT = Path("/opt/afdb-toolkit")
SOURCE_PACKAGE_CONTAINER_PATH = Path("/run/bspp/source-package.tar")
LEGACY_SOURCE_PACKAGE_CONTAINER_PATH = Path("/run/afcdb/source-package.tar")
TOOLKIT_PACKAGE_CONTAINER_PATH = Path("/run/bspp/toolkit-package.tar")
LEGACY_TOOLKIT_PACKAGE_CONTAINER_PATH = Path("/run/afcdb/toolkit-package.tar")
QUALIFICATION_CONTAINER_PATH = Path("/run/bspp/runtime-qualification.json")
RUNTIME_IPSAE_BINARY_CONTAINER_PATH = Path("/run/bspp/runtime-ipsae/ipsae_cpp")
LEGACY_RUNTIME_IPSAE_BINARY_CONTAINER_PATH = Path("/run/afcdb/runtime-ipsae/ipsae_cpp")
PIXI_PYTHON_PATH = Path("/opt/bspp-orchestration-env/.pixi/envs/default/bin/python")
LEGACY_PIXI_PYTHON_PATH = Path("/opt/afcdb-orchestration-env/.pixi/envs/default/bin/python")
AWS_SHARED_CREDENTIALS_CONTAINER_ROOT = Path("/workspace/bspp-aws")


@dataclass(frozen=True)
class WorkflowRenderContext:
    """Inputs outside the RunSpec needed to render runnable workflow jobs."""

    materialized_runspec: Path
    source_bundle_path: Path | None
    control_materialized_runspec: Path | None = None
    control_runtime_qualification_path: Path | None = None
    control_evidence_dir: Path | None = None
    runtime_qualification_path: Path | None = None
    source_package_identity: SourcePackageIdentity | None = None
    toolkit_package_identity: SourcePackageIdentity | None = None
    image_identity: ImageIdentity | None = None
    runtime_ipsae_binary_path: Path | None = None
    runtime_ipsae_binary_sha256: str | None = None
    runtime_qualification_sha256: str | None = None
    runtime_bootstrap_sha256: str | None = None


@dataclass(frozen=True)
class RenderedWorkflowScript:
    """One rendered Control Plane workflow script."""

    step_name: str
    script_path: Path


def render_workflow_step_script(
    spec: RunSpec,
    step: WorkflowStepSpec,
    *,
    context: WorkflowRenderContext,
) -> RenderedWorkflowScript:
    """Render one containerized sbatch script for a workflow step."""
    if (
        spec.run_kind is not None
        and provenance_governed(spec.run_kind)
        and (
            context.source_package_identity is None
            or context.toolkit_package_identity is None
            or context.image_identity is None
        )
    ):
        raise ValueError("governed workflow requires source package, toolkit, and image identities")
    if spec.submission is None:
        msg = "workflow script rendering requires submission.evidence_dir"
        raise ValueError(msg)
    governed = spec.run_kind is not None and provenance_governed(spec.run_kind)
    script_root = context.control_evidence_dir or spec.submission.evidence_dir
    script_path = (
        script_root / "slurm" / f"{step.name}.sbatch"
        if governed and context.control_evidence_dir is not None
        else step.rendered_script or script_root / "slurm" / f"{step.name}.sbatch"
    )
    if not governed:
        script_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = spec.submission.evidence_dir / "slurm-logs"
    # For SSH the evidence dir is remote (not shared with the controller), so the
    # slurm-logs directory must be created on the cluster, not locally.
    if not governed and context.control_evidence_dir == spec.submission.evidence_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    encoded = _render_sbatch_script(spec, step, context=context, log_dir=log_dir).encode()
    if governed:
        script_root.mkdir(parents=True, exist_ok=True)
        try:
            relative = script_path.relative_to(script_root)
        except ValueError as exc:
            raise ValueError("governed rendered script must be below the evidence directory") from exc
        if not script_path.exists():
            write_immutable_descendant(script_root, relative, encoded)
        else:
            existing = stable_read_descendant(script_root, relative)
            if existing != encoded:
                raise ValueError("governed rendered script already exists with different bytes")
    else:
        script_path.write_bytes(encoded)
    return RenderedWorkflowScript(step_name=step.name, script_path=script_path)


def _render_sbatch_script(
    spec: RunSpec,
    step: WorkflowStepSpec,
    *,
    context: WorkflowRenderContext,
    log_dir: Path,
) -> str:
    resource = _resource_for_step(spec, step)
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated workflow step: {step.name}",
        f"# Run ID: {spec.dataset.run_id}",
        f"# Source RunSpec: {context.materialized_runspec}",
        f"# Source SHA256: {spec.source_hash or '<unknown>'}",
        f"#SBATCH --job-name={_job_name(spec, step)}",
        f"#SBATCH --partition={resource.partition}",
        f"#SBATCH --account={spec.cluster.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resource.cpus_per_task}",
        f"#SBATCH --mem={resource.memory}",
        f"#SBATCH --time={resource.time}",
        f"#SBATCH --output={_log_pattern(log_dir, step, stream='out')}",
        f"#SBATCH --error={_log_pattern(log_dir, step, stream='err')}",
    ]
    if resource.gres:
        lines.append(f"#SBATCH --gres={resource.gres}")
    if step.name == "slurm" and resource.array:
        lines.append(f"#SBATCH --array={resource.array}")
    lines.extend(["", "set -euo pipefail", ""])
    payload = _srun_command(spec, step, context=context)
    if spec.run_kind is not None and provenance_governed(spec.run_kind):
        assert spec.submission is not None
        assert spec.workflow is not None
        assert context.runtime_qualification_path is not None
        assert context.runtime_ipsae_binary_path is not None
        step_index = spec.workflow.steps.index(step)
        attempt_dir = spec.submission.evidence_dir / "submissions" / f"{step_index:04d}-{step.name}-attempt-0001"
        lines.extend(
            [
                "set +e",
                payload,
                "BSPP_PAYLOAD_STATUS=$?",
                "set -e",
                render_runtime_result_epilogue(
                    attempt_dir / "expectation.json",
                    attempt_dir / "result.json",
                    runtime_qualification_path=context.runtime_qualification_path,
                    runtime_ipsae_binary_path=context.runtime_ipsae_binary_path,
                    evidence_root=spec.submission.evidence_dir,
                ),
                'exit "$BSPP_PAYLOAD_STATUS"',
                "",
            ]
        )
    else:
        lines.extend([payload, ""])
    return "\n".join(lines)


def _srun_command(spec: RunSpec, step: WorkflowStepSpec, *, context: WorkflowRenderContext) -> str:
    if spec.run_kind is not None and provenance_governed(spec.run_kind):
        assert context.image_identity is not None
        assert context.source_package_identity is not None
        assert context.toolkit_package_identity is not None
        if context.runtime_qualification_path is None:
            raise ValueError("governed workflow requires a mounted identity record")
        if context.runtime_ipsae_binary_path is None or context.runtime_ipsae_binary_sha256 is None:
            raise ValueError("governed workflow requires the qualified runtime-built iPSAE binary")
        if context.runtime_qualification_sha256 is None:
            raise ValueError("governed workflow requires the Runtime Qualification digest")
        runtime_argv = _governed_runtime_argv(spec, step)
        return render_governed_srun(
            context.image_identity,
            mounts=_container_mounts(spec, context),
            bootstrap_args=(
                "run",
                "--identity-record",
                str(QUALIFICATION_CONTAINER_PATH),
                "--source-package",
                str(SOURCE_PACKAGE_CONTAINER_PATH),
                "--toolkit-package",
                str(TOOLKIT_PACKAGE_CONTAINER_PATH),
                "--runtime-ipsae-binary",
                str(RUNTIME_IPSAE_BINARY_CONTAINER_PATH),
                "--expected-runtime-ipsae-sha256",
                context.runtime_ipsae_binary_sha256,
                "--exec-argv-json",
                json.dumps(runtime_argv, separators=(",", ":")),
                "--expected-image-sha256",
                context.image_identity.sha256,
                "--expected-identity-record-sha256",
                context.runtime_qualification_sha256,
            ),
        )
    dev_env = (
        ["env", "BSPP_ORCHESTRATION_DEV_MOUNT=1"] if has_mount_target(spec, ORCHESTRATION_CONTAINER_TARGET) else []
    )
    srun_argv = [
        "srun",
        f"--container-image={spec.container.image}",
        f"--container-mounts={_container_mounts(spec, context)}",
        "--no-container-mount-home",
        *dev_env,
        "/usr/local/bin/entrypoint.sh",
        "bash",
    ]
    return (
        " \\\n  ".join(shlex.quote(part) for part in srun_argv)
        + " <<'BSPP_CONTROL_IN_CONTAINER'\n"
        + _container_script(spec, step, context=context)
        + "\nBSPP_CONTROL_IN_CONTAINER"
    )


def _governed_runtime_argv(spec: RunSpec, step: WorkflowStepSpec) -> tuple[str, ...]:
    """Return one exact, shell-free image Python dispatcher command."""
    command = resolve_workflow_step_command(spec, step)
    lines = command.splitlines()
    calls: list[list[object]] = []
    if step.name == "acceptance-verify-evidence":
        evidence = str(_submission_evidence_dir(spec) / "acceptance")
        calls.append(["acceptance", evidence])
        lines = []
    for line in lines:
        argv = shlex.split(line)
        if not argv:
            continue
        normalized = [str(RUNSPEC_CONTAINER_PATH) if value == "$BSPP_RUNSPEC" else value for value in argv]
        if normalized[0] == "bspp-orchestration-runtime":
            calls.append(["runtime", normalized[1:]])
        elif normalized[:2] == ["$PYTHON_BIN", "-m"]:
            calls.append(["module", normalized[2], normalized[3:]])
        elif normalized[0].startswith("${ORCHESTRATION_ROOT}/"):
            relative = normalized[0].removeprefix("${ORCHESTRATION_ROOT}/")
            calls.append(["exec", relative, normalized[1:]])
        elif normalized[0] == "echo":
            calls.append(["echo", " ".join(normalized[1:])])
        else:
            raise ValueError(f"governed workflow step {step.name!r} lacks a shell-free command mapping")
    launcher = "\n".join(
        (
            "import json, os, runpy, subprocess, sys",
            "source, toolkit, encoded = sys.argv[1:4]",
            "sys.path[:0] = [source + '/packages/orchestration-contract/src',",
            "    source + '/packages/orchestration-runtime/src', toolkit]",
            "from bspp.orchestration.runtime.cli import cli",
            "for call in json.loads(encoded):",
            "    kind = call[0]",
            "    if kind == 'runtime':",
            "        cli.main(args=call[1], standalone_mode=False)",
            "    elif kind == 'module':",
            "        sys.argv = [call[1], *call[2]]",
            "        runpy.run_module(call[1], run_name='__main__')",
            "    elif kind == 'exec':",
            "        values = [os.environ.get('SLURM_CPUS_PER_TASK', '') "
            "if v == '$SLURM_CPUS_PER_TASK' else v for v in call[2]]",
            "        subprocess.run([source + '/' + call[1], *values], check=True)",
            "    elif kind == 'echo':",
            "        print(call[1])",
            "    elif kind == 'acceptance':",
            "        from pathlib import Path",
            "        from bspp.orchestration.runtime.validation.acceptance_evidence import (",
            "            verify_acceptance_evidence, write_acceptance_evidence_report)",
            "        root = Path(call[1])",
            "        report = verify_acceptance_evidence(",
            "            parity_report_path=root/'tar_payload_parity'/'tar_payload_parity_report.json',",
            "            semantic_report_path=root/'semantic_acceptance'/'semantic_acceptance_summary.json')",
            "        write_acceptance_evidence_report(report, root/'verify_evidence')",
            "        if not report.ok: raise SystemExit(1)",
        )
    )
    return (
        "/usr/bin/python3",
        "-I",
        "-S",
        "-c",
        launcher,
        "{BSPP_SOURCE_ROOT}",
        "{BSPP_TOOLKIT_ROOT}",
        json.dumps(calls, separators=(",", ":")),
    )


def _container_script(spec: RunSpec, step: WorkflowStepSpec, *, context: WorkflowRenderContext) -> str:
    source_bundle = str(context.source_bundle_path) if context.source_bundle_path is not None else ""
    runtime_qualification = (
        str(context.runtime_qualification_path) if context.runtime_qualification_path is not None else ""
    )
    lines = [
        "set -euo pipefail",
        "",
        f"BSPP_RUNSPEC={shlex.quote(str(RUNSPEC_CONTAINER_PATH))}",
        f"BSPP_SOURCE_BUNDLE={shlex.quote(source_bundle)}",
        f"BSPP_RUNTIME_QUALIFICATION={shlex.quote(runtime_qualification)}",
        f"ORCHESTRATION_ROOT={shlex.quote(str(ORCHESTRATION_CONTAINER_ROOT))}",
        f"TOOLKIT_ROOT={shlex.quote(str(TOOLKIT_CONTAINER_ROOT))}",
        f'PYTHON_BIN="${{BSPP_ORCHESTRATION_PYTHON:-{PIXI_PYTHON_PATH}}}"',
        'fail() { echo "BSPP workflow step failed: $*" >&2; exit 127; }',
        'require_executable() { [[ -x "$1" ]] || fail "missing executable $2: $1"; }',
        'require_file() { [[ -f "$1" ]] || fail "missing file $2: $1"; }',
        "",
        'require_file "$BSPP_RUNSPEC" "materialized RunSpec"',
        'require_executable "$PYTHON_BIN" "Python runtime"',
        'if [[ -n "$BSPP_SOURCE_BUNDLE" ]]; then',
        '  require_file "$BSPP_SOURCE_BUNDLE" "Source Bundle"',
        '  SOURCE_WORKDIR="${SLURM_TMPDIR:-/tmp}/bspp-orchestration-source"',
        '  rm -rf "$SOURCE_WORKDIR"',
        '  mkdir -p "$SOURCE_WORKDIR"',
        '  tar -xaf "$BSPP_SOURCE_BUNDLE" -C "$SOURCE_WORKDIR"',
        '  ORCHESTRATION_ROOT="$SOURCE_WORKDIR"',
        "fi",
        (
            'export PYTHONPATH="${TOOLKIT_ROOT}:${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:'
            "${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src"
            '${PYTHONPATH:+:${PYTHONPATH}}"'
        ),
        (
            'bspp-orchestration-runtime() { "$PYTHON_BIN" -c '
            '"from bspp.orchestration.runtime.cli import cli; cli()" "$@"; }'
        ),
        "",
        resolve_workflow_step_command(spec, step),
    ]
    if _has_aws_secret_ref(spec):
        aws_env_lines = [
            f"AWS_SHARED_CREDENTIALS_FILE={shlex.quote(str(AWS_SHARED_CREDENTIALS_CONTAINER_ROOT / 'credentials'))}",
            f"AWS_CONFIG_FILE={shlex.quote(str(AWS_SHARED_CREDENTIALS_CONTAINER_ROOT / 'config'))}",
        ]
        aws_profile = _aws_profile_from_spec(spec)
        if aws_profile is not None:
            aws_env_lines.append(f"AWS_PROFILE={shlex.quote(aws_profile)}")
            aws_env_lines.append("export AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE AWS_PROFILE")
        else:
            aws_env_lines.append("export AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE")
        lines[5:5] = aws_env_lines
    return "\n".join(lines)


def resolve_workflow_step_command(spec: RunSpec, step: WorkflowStepSpec, *, legacy: bool = False) -> str:
    """Resolve the pure legacy command body for one already-validated step.

    ``legacy=True`` reproduces the pre-rename CLI/env names for historical
    V1/V2 renderer contracts.
    """
    command = _resolve_workflow_step_command_impl(spec, step)
    if legacy:
        command = (
            command.replace("bspp-orchestration-runtime", "afcdb-orchestration-runtime")
            .replace("BSPP_", "AFCDB_")
            .replace("bspp.orchestration.runtime", "afcdb.orchestration.runtime")
        )
    return command


def _resolve_workflow_step_command_impl(spec: RunSpec, step: WorkflowStepSpec) -> str:
    if step.name == "preflight":
        return 'bspp-orchestration-runtime runspec preflight "$BSPP_RUNSPEC" --write-report --strict'
    if step.name == "recipe":
        return (
            'bspp-orchestration-runtime inputs coverage --runspec "$BSPP_RUNSPEC" --write-report\n'
            'bspp-orchestration-runtime runspec render-recipe "$BSPP_RUNSPEC" --execute'
        )
    if step.name == "preprocess":
        return (
            'bspp-orchestration-runtime inputs prepare --runspec "$BSPP_RUNSPEC" '
            "--execute --strict-secrets --write-report\n"
            'bspp-orchestration-runtime runspec render-preprocess "$BSPP_RUNSPEC" --execute --no-allowlists'
        )
    if step.name == "slurm":
        return 'bspp-orchestration-runtime worker archive-task --runspec "$BSPP_RUNSPEC"'
    if step.name == "analysis-finalize":
        return _analysis_finalizer_command(spec)
    if step.name == "acceptance-tar-payload-parity":
        return _tar_payload_parity_command(spec)
    if step.name == "acceptance-semantic":
        return _semantic_acceptance_command(spec)
    if step.name == "acceptance-verify-evidence":
        return _acceptance_verify_command(spec)
    if step.name in {"download-manifest", "download"}:
        return 'bspp-orchestration-runtime inputs stage-archives --runspec "$BSPP_RUNSPEC" --execute --write-report'
    if step.name in {"extract", "extract-local"}:
        return _extract_command(spec)
    if step.name == "aggregate":
        return 'bspp-orchestration-runtime runspec validate-archive-output "$BSPP_RUNSPEC" --write-report --strict'
    if step.name == "upload-s3":
        return _upload_s3_command(spec)
    if step.name == "upload-gcs":
        return _upload_gcs_command(spec)
    if step.name == "upload-gcs-direct":
        return _upload_gcs_direct_command(spec)
    if step.name in {"cleanup", "cleanup-local"}:
        return 'bspp-orchestration-runtime runspec plan-cleanup "$BSPP_RUNSPEC" --execute'
    if step.name in {"status", "rename-afid"}:
        return 'bspp-orchestration-runtime runspec validate "$BSPP_RUNSPEC"'
    msg = f"unsupported workflow step for Control Plane rendering: {step.name}"
    raise ValueError(msg)


def _analysis_finalizer_command(spec: RunSpec) -> str:
    metadata = spec.analysis_metadata
    if not metadata.enabled:
        return 'echo "analysis metadata finalizer disabled by RunSpec"'
    if metadata.csv_path is None or metadata.parquet_path is None or metadata.selected_ids_path is None:
        msg = "analysis-finalize requires analysis metadata csv, parquet, and selected ids paths"
        raise ValueError(msg)
    return " ".join(
        (
            '"$PYTHON_BIN"',
            "-m",
            "bspp.orchestration.runtime.postprocessing.analysis_finalizer",
            "--csv",
            shlex.quote(str(metadata.csv_path)),
            "--parquet",
            shlex.quote(str(metadata.parquet_path)),
            "--selected-ids",
            shlex.quote(str(metadata.selected_ids_path)),
        )
    )


def _tar_payload_parity_command(spec: RunSpec) -> str:
    if spec.acceptance is None or spec.acceptance.baseline_output_dir is None:
        msg = "acceptance-tar-payload-parity requires acceptance.baseline_output_dir"
        raise ValueError(msg)
    evidence_dir = _submission_evidence_dir(spec)
    parts = [
        _orchestration_script("slurm-tar-payload-parity.sh"),
        "--baseline-dir",
        shlex.quote(str(spec.acceptance.baseline_output_dir)),
        "--candidate-dir",
        shlex.quote(str(spec.paths.output_dir)),
        "--relative-dir",
        "local_tars",
        "--exclude",
        "metadata/",
        "--match-mode",
        spec.acceptance.tar_payload_match_mode,
        "--workers",
        '"$SLURM_CPUS_PER_TASK"',
        "--write-report",
        shlex.quote(str(evidence_dir / "acceptance" / "tar_payload_parity")),
        "--strict",
    ]
    if spec.acceptance.baseline_run_name is not None:
        parts.extend(("--baseline-run-name", shlex.quote(spec.acceptance.baseline_run_name)))
    if spec.acceptance.candidate_run_name is not None:
        parts.extend(("--candidate-run-name", shlex.quote(spec.acceptance.candidate_run_name)))
    if spec.acceptance.payload_sample_count is not None:
        parts.extend(("--payload-sample-count", str(spec.acceptance.payload_sample_count)))
    return " ".join(parts)


def _semantic_acceptance_command(spec: RunSpec) -> str:
    if spec.acceptance is None or spec.acceptance.baseline_output_dir is None:
        msg = "acceptance-semantic requires acceptance.baseline_output_dir"
        raise ValueError(msg)
    evidence_dir = _submission_evidence_dir(spec)
    parts = [
        _orchestration_script("slurm-semantic-acceptance.sh"),
        "--baseline-dir",
        shlex.quote(str(spec.acceptance.baseline_output_dir)),
        "--candidate-dir",
        shlex.quote(str(spec.paths.output_dir)),
        "--write-report",
        shlex.quote(str(evidence_dir / "acceptance" / "semantic_acceptance")),
        "--strict",
    ]
    for option, value in (
        ("--expected-tar-count", spec.validation.expected_tar_count),
        ("--expected-local-tars-rows", spec.validation.expected_local_tars_rows),
        ("--expected-failed-rows", spec.validation.expected_failed_rows),
        ("--expected-analysis-rows", spec.validation.expected_analysis_rows),
        ("--expected-selected-ids", spec.validation.expected_selected_ids),
    ):
        if value is not None:
            parts.extend((option, str(value)))
    if not spec.acceptance.candidate_parquet_required:
        parts.append("--candidate-parquet-optional")
    if not spec.acceptance.compare_failed_sets:
        parts.append("--no-compare-failed-sets")
    return " ".join(parts)


def _orchestration_script(script_name: str) -> str:
    return f'"${{ORCHESTRATION_ROOT}}/containers/scripts/{script_name}"'


def _acceptance_verify_command(spec: RunSpec) -> str:
    evidence = _submission_evidence_dir(spec) / "acceptance"
    return "\n".join(
        (
            "\"$PYTHON_BIN\" - <<'BSPP_VERIFY_ACCEPTANCE'",
            "from pathlib import Path",
            "from bspp.orchestration.runtime.validation.acceptance_evidence import (",
            "    verify_acceptance_evidence,",
            "    write_acceptance_evidence_report,",
            ")",
            f"root = Path({str(evidence)!r})",
            "report = verify_acceptance_evidence(",
            "    parity_report_path=root / 'tar_payload_parity' / 'tar_payload_parity_report.json',",
            "    semantic_report_path=root / 'semantic_acceptance' / 'semantic_acceptance_summary.json',",
            ")",
            "write_acceptance_evidence_report(report, root / 'verify_evidence')",
            "raise SystemExit(0 if report.ok else 1)",
            "BSPP_VERIFY_ACCEPTANCE",
        )
    )


def _extract_command(spec: RunSpec) -> str:
    return " ".join(
        (
            "bspp-orchestration-runtime",
            "extract",
            "--staging-dir",
            shlex.quote(str(spec.paths.staging_dir)),
            "--output-dir",
            shlex.quote(str(spec.paths.output_dir)),
            "--parallel",
            str(spec.worker.workers),
            "--keep-archives",
        )
    )


def _upload_s3_command(spec: RunSpec) -> str:
    selection = _data_placement_selection(spec, field_name="s3", step_name="upload-s3")
    evidence_dir = _data_placement_stage_dir(spec, "s3")
    return " ".join(
        (
            "bspp-orchestration-runtime",
            "upload-s3",
            "--dataset",
            shlex.quote(spec.dataset.name),
            "--output-base",
            shlex.quote(str(spec.paths.output_dir.parent)),
            "--dataset-output-dir",
            shlex.quote(str(spec.paths.output_dir)),
            "--tool",
            selection.tool,
            "--data-dir",
            shlex.quote(str(evidence_dir)),
            "--write-evidence",
            shlex.quote(str(evidence_dir / "data-placement.json")),
            "--s3-destination-prefix",
            shlex.quote(spec.storage.s3_output_prefix),
            "--execute",
        )
    )


def _upload_gcs_command(spec: RunSpec) -> str:
    selection = _data_placement_selection(spec, field_name="gcs", step_name="upload-gcs")
    evidence_dir = _data_placement_stage_dir(spec, "gcs")
    destination = spec.storage.gcs_destination_prefix or ""
    return " ".join(
        (
            "bspp-orchestration-runtime",
            "upload-gcs",
            "--dataset",
            shlex.quote(spec.dataset.name),
            "--data-dir",
            shlex.quote(str(evidence_dir)),
            "--tool",
            selection.tool,
            "--write-evidence",
            shlex.quote(str(evidence_dir / "data-placement.json")),
            "--s3-source-prefix",
            shlex.quote(spec.storage.s3_output_prefix),
            "--gcs-destination-prefix",
            shlex.quote(destination),
            "--execute",
        )
    )


def _upload_gcs_direct_command(spec: RunSpec) -> str:
    _data_placement_selection(spec, field_name="gcs_direct", step_name="upload-gcs-direct")
    evidence_dir = _data_placement_stage_dir(spec, "gcs-direct")
    destination = spec.storage.gcs_destination_prefix or ""
    return " ".join(
        (
            "bspp-orchestration-runtime",
            "upload-gcs-direct",
            "--dataset",
            shlex.quote(spec.dataset.name),
            "--output-base",
            shlex.quote(str(spec.paths.output_dir.parent)),
            "--dataset-output-dir",
            shlex.quote(str(spec.paths.output_dir)),
            "--write-evidence",
            shlex.quote(str(evidence_dir / "data-placement.json")),
            "--gcs-destination-prefix",
            shlex.quote(destination),
            "--execute",
        )
    )


def _data_placement_selection(spec: RunSpec, *, field_name: str, step_name: str) -> DataMoverSelectionSpec:
    if field_name == "s3":
        selection = spec.data_placement.s3
    elif field_name == "gcs":
        selection = spec.data_placement.gcs
    elif field_name == "gcs_direct":
        selection = spec.data_placement.gcs_direct
    else:
        msg = f"unsupported data placement field {field_name!r}"
        raise ValueError(msg)
    if selection is None:
        msg = f"workflow step {step_name} requires data_placement.{field_name}"
        raise ValueError(msg)
    return selection


def _data_placement_stage_dir(spec: RunSpec, stage_name: str) -> Path:
    return _submission_evidence_dir(spec) / "data-placement" / stage_name


def _resource_for_step(spec: RunSpec, step: WorkflowStepSpec) -> SlurmResources:
    resource_key = {
        "preflight": "control_cpu",
        "recipe": "control_cpu",
        "preprocess": "control_cpu",
        "slurm": "gpu_worker",
        "analysis-finalize": "analysis_finalize",
        "acceptance-tar-payload-parity": "acceptance_tar_payload_parity",
        "acceptance-semantic": "acceptance_semantic",
        "acceptance-verify-evidence": "control_cpu",
    }.get(step.name, step.name.replace("-", "_"))
    resource = spec.resources.get(resource_key) or spec.resources.get("gpu_worker")
    if resource is None:
        msg = f"workflow step {step.name!r} requires Slurm resources {resource_key!r} or 'gpu_worker'"
        raise ValueError(msg)
    return resource


def _submission_evidence_dir(spec: RunSpec) -> Path:
    if spec.submission is None:
        msg = "workflow script rendering requires submission.evidence_dir"
        raise ValueError(msg)
    return spec.submission.evidence_dir


def _container_mounts(spec: RunSpec, context: WorkflowRenderContext) -> str:
    if spec.run_kind is not None and provenance_governed(spec.run_kind):
        assert context.source_package_identity is not None
        assert context.toolkit_package_identity is not None
        assert context.runtime_qualification_path is not None
        assert context.runtime_ipsae_binary_path is not None
        mounts = [
            (str(context.source_package_identity.package_path), f"{SOURCE_PACKAGE_CONTAINER_PATH}:ro"),
            (str(context.toolkit_package_identity.package_path), f"{TOOLKIT_PACKAGE_CONTAINER_PATH}:ro"),
            (str(context.materialized_runspec), f"{RUNSPEC_CONTAINER_PATH}:ro"),
            (str(context.runtime_qualification_path), f"{QUALIFICATION_CONTAINER_PATH}:ro"),
            (str(context.runtime_ipsae_binary_path), f"{RUNTIME_IPSAE_BINARY_CONTAINER_PATH}:ro"),
        ]
        protected_sources = {
            context.source_package_identity.package_path,
            context.toolkit_package_identity.package_path,
            context.materialized_runspec,
            context.runtime_qualification_path,
            context.runtime_ipsae_binary_path,
        }
        protected_targets = {
            SOURCE_PACKAGE_CONTAINER_PATH,
            TOOLKIT_PACKAGE_CONTAINER_PATH,
            RUNSPEC_CONTAINER_PATH,
            QUALIFICATION_CONTAINER_PATH,
            RUNTIME_IPSAE_BINARY_CONTAINER_PATH,
        }
        ambient = {spec.paths.orchestration_repo, spec.paths.afdb_toolkit_repo, spec.paths.legacy_repo}
        if _has_aws_secret_ref(spec):
            ambient.add(Path.home() / ".aws")
        for mount in spec.container.mounts:
            source = _canonical_governed_mount_path(mount.source)
            target_path = _canonical_governed_mount_path(mount.target, require_absolute=True)
            canonical_ambient = {_canonical_governed_mount_path(path) for path in ambient if path is not None}
            if source in canonical_ambient:
                continue
            if any(_paths_overlap(source, protected) for protected in canonical_ambient):
                raise ValueError("governed mount overlaps protected ambient orchestration or toolkit checkout")
            writable = _governed_mount_is_writable(spec, source)
            if writable and (
                any(
                    _paths_overlap(source, _canonical_governed_mount_path(protected)) for protected in protected_sources
                )
                or any(_paths_overlap(target_path, protected) for protected in protected_targets)
            ):
                raise ValueError("governed writable mount overlaps protected identity input")
            target = str(target_path) if writable else f"{target_path}:ro"
            mounts.append((str(source), target))
        if _has_aws_secret_ref(spec):
            mounts.extend(
                (
                    (
                        str(Path.home() / ".aws" / "credentials"),
                        f"{AWS_SHARED_CREDENTIALS_CONTAINER_ROOT / 'credentials'}:ro",
                    ),
                    (str(Path.home() / ".aws" / "config"), f"{AWS_SHARED_CREDENTIALS_CONTAINER_ROOT / 'config'}:ro"),
                )
            )
        return ",".join(f"{source}:{target}" for source, target in mounts)
    mounts = [(str(mount.source), str(mount.target)) for mount in spec.container.mounts]
    mounts.append((str(context.materialized_runspec), str(RUNSPEC_CONTAINER_PATH)))
    if _has_aws_secret_ref(spec):
        mounts.append((str(Path.home() / ".aws"), str(AWS_SHARED_CREDENTIALS_CONTAINER_ROOT)))
    if context.source_bundle_path is not None:
        bundle_root = context.source_bundle_path.parent
        mounts.append((str(bundle_root), str(bundle_root)))
    if context.runtime_qualification_path is not None:
        qualification_root = context.runtime_qualification_path.parent
        mounts.append((str(qualification_root), str(qualification_root)))
    return ",".join(f"{source}:{target}" for source, target in _dedupe_mounts(mounts))


def _dedupe_mounts(mounts: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for mount in mounts:
        if mount in seen:
            continue
        seen.add(mount)
        result.append(mount)
    return tuple(result)


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    return _path_contains(first, second) or _path_contains(second, first)


def _canonical_governed_mount_path(path: Path, *, require_absolute: bool = False) -> Path:
    if ".." in path.parts:
        raise ValueError("governed mount path must not contain traversal aliases")
    lexical = path.absolute()
    if require_absolute and not path.is_absolute():
        raise ValueError("governed mount target must be absolute")
    resolved = lexical.resolve(strict=False)
    if resolved != lexical:
        raise ValueError("governed mount path must not use symlink or lexical aliases")
    return resolved


def _governed_mount_is_writable(spec: RunSpec, source: Path) -> bool:
    writable_paths = (spec.paths.staging_dir, spec.paths.output_dir, spec.paths.log_dir, spec.paths.recipe_dir)
    return any(
        path is not None and (_path_contains(source, path) or _path_contains(path, source)) for path in writable_paths
    )


def _has_aws_secret_ref(spec: RunSpec) -> bool:
    return any(ref.scheme == "aws" for ref in spec.secret_refs().values())


def _aws_profile_from_spec(spec: RunSpec) -> str | None:
    for ref in spec.secret_refs().values():
        if ref.scheme == "aws":
            if ref.target in {"auto", "profile:auto"}:
                return None
            return ref.target.removeprefix("profile:") or "default"
    return None


def _job_name(spec: RunSpec, step: WorkflowStepSpec) -> str:
    return f"bspp_{spec.dataset.run_id}_{step.name}".replace("-", "_")


def _log_pattern(log_dir: Path, step: WorkflowStepSpec, *, stream: str) -> Path:
    token = "%A_%a" if step.name == "slurm" else "%j"
    return log_dir / f"{step.name.replace('-', '_')}_{token}.{stream}"


__all__ = [
    "BAKED_TOOLKIT_CONTAINER_ROOT",
    "ORCHESTRATION_CONTAINER_ROOT",
    "PIXI_PYTHON_PATH",
    "RUNSPEC_CONTAINER_PATH",
    "TOOLKIT_CONTAINER_ROOT",
    "RenderedWorkflowScript",
    "WorkflowRenderContext",
    "render_workflow_step_script",
    "resolve_workflow_step_command",
]
