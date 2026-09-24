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

"""Workstation-side submission for scheduled folding benchmark validation.

This module renders and submits one bounded Slurm job that runs the runtime
``benchmark validate-run`` worker inside the runtime container, monitors it to a
terminal Slurm state, and fetches only the worker's small ``summary.json`` back
to the workstation. The full ``validation.parquet`` and run artifacts stay
cluster-side.

It deliberately imports only the standard library and
``bspp.orchestration.control.*`` so the Control Plane import-boundary gate
remains green (no runtime, numpy, or pyarrow imports).
"""

from __future__ import annotations

import hashlib
import json
import shlex
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.control.profiles import ResolvedClusterProfile
from bspp.orchestration.control.transport import (
    TERMINAL_SLURM_STATES,
    CommandRunner,
    RemoteSlurmTransport,
    SlurmJobState,
    default_command_runner,
)

_BENCHMARK_SCHEDULING_CLASS = "control_cpu"
_MAX_SUMMARY_BYTES = 1024 * 1024  # 1 MiB bounded summary read
_JOB_NAME_PREFIX = "bspp_validate_run_"
_ORCHESTRATION_CONTAINER_TARGET = "/workspace/bspp-orchestration"
_AWS_CONTAINER_ROOT = "/workspace/bspp-aws"


@dataclass(frozen=True)
class RenderedBenchmarkValidateSubmission:
    """One rendered benchmark-validation Slurm submission."""

    script_path: Path
    cluster_script_path: Path
    summary_path: Path
    job_name: str
    action_command: str


@dataclass(frozen=True)
class BenchmarkValidateResult:
    """Bounded validation evidence returned to the workstation."""

    job_id: str
    summary: Mapping[str, object]
    accepted: bool

    def render_json(self) -> str:
        """Render the bounded summary as stable JSON."""
        return json.dumps(self.summary, sort_keys=True)


def render_benchmark_validate_submission(
    *,
    cluster_profile: ResolvedClusterProfile,
    run_dir: Path,
    suite: Path,
    index: Path,
    corpus: str,
    fingerprint: str,
    output_dir: Path,
    script_path: Path,
    aws_profile: str | None = None,
) -> RenderedBenchmarkValidateSubmission:
    """Render a bounded Slurm submission for the runtime validate-run worker."""
    if not fingerprint:
        raise ValueError("benchmark validation fingerprint must be non-empty")
    if not corpus:
        raise ValueError("benchmark validation corpus must be non-empty")
    if aws_profile is not None and not aws_profile:
        raise ValueError("benchmark validation AWS profile must be non-empty when supplied")
    for name, path in (("run-dir", run_dir), ("suite", suite), ("index", index), ("output-dir", output_dir)):
        if not path.is_absolute():
            raise ValueError(f"benchmark validation {name} must be an absolute cluster path")
    resource = cluster_profile.resources.get(_BENCHMARK_SCHEDULING_CLASS)
    if resource is None:
        raise ValueError(f"Cluster Profile {cluster_profile.name!r} requires resources.{_BENCHMARK_SCHEDULING_CLASS}")
    credential_mounts = cluster_profile.postprocessing_credential_mounts
    if credential_mounts is None:
        raise ValueError(
            "benchmark validation requires Cluster Profile postprocessing_credential_mounts "
            "with both cluster-visible shared-profile files"
        )

    job_name = _job_name(fingerprint)
    token = job_name.removeprefix(_JOB_NAME_PREFIX)
    mounts = _dedupe_mounts(
        [
            (str(run_dir), str(run_dir), False),
            (str(suite), str(suite), False),
            (str(index), str(index), False),
            (str(output_dir), str(output_dir), True),
            (str(cluster_profile.orchestration_repo), _ORCHESTRATION_CONTAINER_TARGET, False),
            (credential_mounts.aws_shared_credentials_file, f"{_AWS_CONTAINER_ROOT}/credentials", False),
            (credential_mounts.aws_config_file, f"{_AWS_CONTAINER_ROOT}/config", False),
        ]
    )
    mounts_arg = ",".join(_render_mount(mount) for mount in mounts)
    action_command = " ".join(
        (
            "bspp-orchestration-runtime",
            "benchmark",
            "validate-run",
            "--run-dir",
            shlex.quote(str(run_dir)),
            "--suite",
            shlex.quote(str(suite)),
            "--index",
            shlex.quote(str(index)),
            "--corpus",
            shlex.quote(corpus),
            "--fingerprint",
            shlex.quote(fingerprint),
            "--output-dir",
            shlex.quote(str(output_dir)),
        )
    )
    log_dir = output_dir / "slurm-logs"
    srun = [
        "srun",
        f"--container-image={cluster_profile.image}",
        f"--container-mounts={mounts_arg}",
        "--no-container-mount-home",
        "env",
        "BSPP_INSTALL_MODE_SKIP=1",
        "/usr/local/bin/entrypoint.sh",
        "bash",
    ]
    srun_line = " \\\n  ".join(shlex.quote(part) for part in srun)
    heredoc = [
        f"AWS_SHARED_CREDENTIALS_FILE={shlex.quote(f'{_AWS_CONTAINER_ROOT}/credentials')}",
        f"AWS_CONFIG_FILE={shlex.quote(f'{_AWS_CONTAINER_ROOT}/config')}",
        "export AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE",
    ]
    if aws_profile is not None:
        heredoc.extend((f"AWS_PROFILE={shlex.quote(aws_profile)}", "export AWS_PROFILE"))
    heredoc.append(action_command)
    lines = [
        "#!/usr/bin/env bash",
        "# BSPP benchmark validation",
        f"# Job name: {job_name}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resource.partition}",
        f"#SBATCH --account={cluster_profile.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resource.cpus_per_task}",
        f"#SBATCH --mem={resource.memory}",
        f"#SBATCH --time={resource.time}",
        f"#SBATCH --output={log_dir / f'validate_run.{token}.%j.out'}",
        f"#SBATCH --error={log_dir / f'validate_run.{token}.%j.err'}",
        "",
        "set -euo pipefail",
        "",
        srun_line + " <<'BSPP_BENCHMARK_VALIDATE'",
        *heredoc,
        "BSPP_BENCHMARK_VALIDATE",
        "",
    ]
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines), encoding="utf-8")
    cluster_script_path = output_dir / "validate-run.sbatch" if cluster_profile.transport == "ssh" else script_path
    return RenderedBenchmarkValidateSubmission(
        script_path=script_path,
        cluster_script_path=cluster_script_path,
        summary_path=output_dir / "summary.json",
        job_name=job_name,
        action_command=action_command,
    )


def submit_benchmark_validation(
    *,
    cluster_profile: ResolvedClusterProfile,
    run_dir: Path,
    suite: Path,
    index: Path,
    corpus: str,
    fingerprint: str,
    output_dir: Path,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 86400.0,
    runner: CommandRunner = default_command_runner,
    sleeper: Callable[[float], None] = time.sleep,
    aws_profile: str | None = None,
) -> BenchmarkValidateResult:
    """Render, submit, monitor, and fetch bounded evidence for one validation."""
    job_name = _job_name(fingerprint)
    local_script_path = _local_script_path(cluster_profile, output_dir, job_name=job_name)
    local_script_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_benchmark_validate_submission(
        cluster_profile=cluster_profile,
        run_dir=run_dir,
        suite=suite,
        index=index,
        corpus=corpus,
        fingerprint=fingerprint,
        output_dir=output_dir,
        script_path=local_script_path,
        aws_profile=aws_profile,
    )
    transport = RemoteSlurmTransport(
        kind=cluster_profile.transport,
        ssh_target=cluster_profile.ssh_target,
        runner=runner,
    )
    # The rendered script points --output/--error at output_dir/slurm-logs, so
    # that directory must exist before submission on either transport; Slurm
    # fails the job at startup when it cannot open its log paths.
    log_dir = output_dir / "slurm-logs"
    if cluster_profile.transport == "ssh":
        log_dir_result = transport.command(("mkdir", "-p", str(log_dir)))
        if log_dir_result.returncode != 0:
            detail = log_dir_result.stderr.strip() or log_dir_result.stdout.strip() or "mkdir failed"
            raise ValueError(f"failed to create benchmark validation Slurm log directory {log_dir}: {detail}")
        transport.stage_immutable_artifact(
            rendered.script_path,
            str(rendered.cluster_script_path),
            expected_sha256=_file_sha256(rendered.script_path),
            staging_token=f"benchmark-validate-{job_name.removeprefix(_JOB_NAME_PREFIX)}",
        )
        submission = transport.submit_script(rendered.cluster_script_path)
    else:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"failed to create benchmark validation Slurm log directory {log_dir}: {exc}") from exc
        submission = transport.submit_script(rendered.script_path)

    state = _wait_for_terminal_state(
        transport,
        submission.job_id,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        sleeper=sleeper,
    )
    if state.state != "COMPLETED" or state.exit_code != "0:0":
        raise ValueError(
            f"benchmark validation job {submission.job_id} ended in state {state.state!r} "
            f"with exit code {state.exit_code!r}; expected COMPLETED with exit code 0:0"
        )
    summary = _fetch_summary(transport, rendered.summary_path)
    accepted = _summary_accepted(summary)
    return BenchmarkValidateResult(job_id=submission.job_id, summary=summary, accepted=accepted)


def _wait_for_terminal_state(
    transport: RemoteSlurmTransport,
    job_id: str,
    *,
    poll_interval_seconds: float,
    timeout_seconds: float,
    sleeper: Callable[[float], None],
) -> SlurmJobState:
    deadline = time.monotonic() + timeout_seconds
    while True:
        observation = transport.query_observation((job_id,), require_exact_terminal_exit=True)
        states = observation.selected_states
        if states:
            state = states[0]
            if state.state in TERMINAL_SLURM_STATES:
                return state
        if time.monotonic() >= deadline:
            raise ValueError(f"timed out waiting for benchmark validation job {job_id} to reach a terminal Slurm state")
        sleeper(poll_interval_seconds)


def _fetch_summary(transport: RemoteSlurmTransport, summary_path: Path) -> Mapping[str, object]:
    with tempfile.TemporaryDirectory(prefix="bspp-benchmark-validate-") as tmp_dir:
        local_path = Path(tmp_dir) / "summary.json"
        result = transport.fetch_artifact(str(summary_path), local_path)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "summary fetch failed"
            raise ValueError(f"failed to fetch benchmark validation summary {summary_path}: {detail}")
        if not local_path.is_file() or local_path.is_symlink():
            raise ValueError("benchmark validation summary fetch did not produce a regular file")
        if local_path.stat().st_size > _MAX_SUMMARY_BYTES:
            raise ValueError("benchmark validation summary exceeds the bounded read limit")
        try:
            payload = json.loads(local_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"benchmark validation summary is missing or invalid: {summary_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("benchmark validation summary must be a JSON object")
        return payload


def _summary_accepted(summary: Mapping[str, object]) -> bool:
    passed = _int_field(summary, "passed_count") if "passed_count" in summary else _int_field(summary, "passed")
    case_count = _int_field(summary, "case_count") if "case_count" in summary else len(_list_field(summary, "cases"))
    return passed == case_count


def _int_field(summary: Mapping[str, object], name: str) -> int:
    value = summary.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"benchmark validation summary field {name!r} must be an integer")
    return value


def _list_field(summary: Mapping[str, object], name: str) -> list[object]:
    value = summary.get(name)
    if not isinstance(value, list):
        raise ValueError(f"benchmark validation summary field {name!r} must be a list")
    return value


def _job_name(fingerprint: str) -> str:
    return _JOB_NAME_PREFIX + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]


def _local_script_path(
    cluster_profile: ResolvedClusterProfile,
    output_dir: Path,
    *,
    job_name: str,
) -> Path:
    if cluster_profile.transport == "ssh":
        return Path.cwd() / ".bspp-benchmark" / f"{job_name}.sbatch"
    return output_dir / "validate-run.sbatch"


def _render_mount(mount: tuple[str, str, bool]) -> str:
    source, target, writable = mount
    rendered = f"{source}:{target}"
    if not writable:
        rendered += ":ro"
    return rendered


def _dedupe_mounts(mounts: list[tuple[str, str, bool]]) -> tuple[tuple[str, str, bool], ...]:
    by_source: dict[str, tuple[str, str, bool]] = {}
    for source, target, writable in mounts:
        existing = by_source.get(source)
        if existing is None or (writable and not existing[2]):
            by_source[source] = (source, target, writable)
    return tuple(by_source.values())


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


__all__ = [
    "BenchmarkValidateResult",
    "RenderedBenchmarkValidateSubmission",
    "render_benchmark_validate_submission",
    "submit_benchmark_validation",
]
