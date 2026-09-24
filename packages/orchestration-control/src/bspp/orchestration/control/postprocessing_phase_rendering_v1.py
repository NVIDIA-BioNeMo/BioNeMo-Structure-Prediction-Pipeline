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

"""Frozen deterministic renderer for historical postprocessing V1 authority."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.postprocessing_runspec import ReadablePostprocessingPhaseRunSpec
from bspp.orchestration.contract.postprocessing_runspec_v1 import HistoricalPostprocessingPhaseRunSpecV1
from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.control.execution_bootstrap import ImageIdentity, render_governed_srun
from bspp.orchestration.control.postprocessing_scheduler_identity import (
    postprocessing_scheduler_correlation_token,
)
from bspp.orchestration.control.workflow_rendering import (
    LEGACY_PIXI_PYTHON_PATH,
    LEGACY_RUNSPEC_CONTAINER_PATH,
    LEGACY_RUNTIME_IPSAE_BINARY_CONTAINER_PATH,
    LEGACY_SOURCE_PACKAGE_CONTAINER_PATH,
    LEGACY_TOOLKIT_PACKAGE_CONTAINER_PATH,
    resolve_workflow_step_command,
)

_POLICY_CONTAINER_PATH = "/run/afcdb/postprocessing-acceptance-policy.json"
_QUALIFICATION_CONTAINER_PATH = "/run/afcdb/runtime-qualification.json"
_PHASE_LEGACY_RUNSPEC_CONTAINER_PATH = "/run/afcdb/postprocessing-phase-runspec.json"


@dataclass(frozen=True)
class PostprocessingRenderInput:
    runspec: ReadablePostprocessingPhaseRunSpec
    legacy_runspec: RunSpec

    @property
    def phase_run_id(self) -> str:
        return self.runspec.phase_run_id

    @property
    def attempt_id(self) -> str:
        return self.runspec.attempt_id


def render_historical_postprocessing_action_script_v1(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> str:
    """Render exactly the frozen V1 renderer contract for V1 authority."""
    if not isinstance(authority.runspec, HistoricalPostprocessingPhaseRunSpecV1):
        raise ValueError("the historical postprocessing renderer accepts only V1 authority")
    return _render_postprocessing_action_script_body(authority, action)


HistoricalPostprocessingRenderer = Callable[[PostprocessingRenderInput, PostprocessingRuntimeAction], str]

HISTORICAL_POSTPROCESSING_RENDERERS: Mapping[int, HistoricalPostprocessingRenderer] = MappingProxyType(
    {1: render_historical_postprocessing_action_script_v1}
)


def historical_postprocessing_renderer(version: int) -> HistoricalPostprocessingRenderer:
    try:
        return HISTORICAL_POSTPROCESSING_RENDERERS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported historical postprocessing renderer contract version: {version}") from exc


def _render_postprocessing_action_script_body(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> str:
    """Render the frozen renderer-contract-v1 body after family validation."""
    cluster_attempt = _cluster_attempt_root(authority.runspec, legacy=True)
    cluster_runspec = cluster_attempt / "legacy-runspec.yaml"
    cluster_phase_runspec = cluster_attempt / "phase-runspec.json"
    cluster_policy = cluster_attempt / "acceptance-policy.json"
    cluster_qualification = cluster_attempt / "runtime-qualification.json"
    evidence_dir = authority.runspec.payload.attempt_paths.evidence_dir
    scheduler_token = postprocessing_scheduler_correlation_token(authority.runspec, action, legacy=True)
    lines = [
        "#!/usr/bin/env bash",
        f"# Postprocessing Phase action: {action.action_id}",
        f"# Phase Run: {authority.phase_run_id}",
        f"# Attempt: {authority.attempt_id}",
        f"#SBATCH --job-name={scheduler_token}",
        f"#SBATCH --comment={scheduler_token}",
        f"#SBATCH --partition={action.resources.partition}",
        f"#SBATCH --account={authority.runspec.cluster.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={action.resources.cpus_per_task}",
        f"#SBATCH --mem={action.resources.memory}",
        f"#SBATCH --time={action.resources.time}",
        f"#SBATCH --output={evidence_dir}/slurm-logs/{action.step_name}.%A_%a.out",
        f"#SBATCH --error={evidence_dir}/slurm-logs/{action.step_name}.%A_%a.err",
    ]
    if action.resources.gres:
        lines.append(f"#SBATCH --gres={action.resources.gres}")
    if action.normalized_array is not None:
        lines.append(f"#SBATCH --array={action.normalized_array}")
    if action.resources.nodelist is not None:
        lines.append(f"#SBATCH --nodelist={action.resources.nodelist}")
    lines.extend(["", "set -euo pipefail", f"mkdir -p {shlex.quote(evidence_dir + '/slurm-logs')}", ""])
    raw_command = _postprocessing_raw_command(authority, action)
    container_body = _container_action_body(
        authority,
        action,
        raw_command=raw_command,
    )
    qualified = authority.runspec.payload.qualified_runtime
    image = ImageIdentity(
        format_version=1,
        policy=qualified.image_policy,
        path=Path(qualified.image_path),
        size_bytes=qualified.image_size_bytes,
        sha256=qualified.image_sha256,
    )
    mount_text = _postprocessing_governed_mounts(
        authority,
        cluster_runspec=cluster_runspec,
        cluster_phase_runspec=cluster_phase_runspec,
        cluster_policy=cluster_policy,
        cluster_qualification=cluster_qualification,
    )
    bootstrap_args = [
        "run",
        "--identity-record",
        _QUALIFICATION_CONTAINER_PATH,
        "--source-package",
        str(LEGACY_SOURCE_PACKAGE_CONTAINER_PATH),
    ]
    if qualified.source_kind == "override":
        bootstrap_args.extend(("--toolkit-package", str(LEGACY_TOOLKIT_PACKAGE_CONTAINER_PATH)))
    bootstrap_args.extend(
        (
            "--runtime-ipsae-binary",
            str(LEGACY_RUNTIME_IPSAE_BINARY_CONTAINER_PATH),
            "--expected-runtime-ipsae-sha256",
            qualified.runtime_ipsae_binary_sha256,
            "--exec-argv-json",
            json.dumps(
                ("/usr/bin/bash", "-c", container_body),
                separators=(",", ":"),
            ),
            "--expected-image-sha256",
            qualified.image_sha256,
            "--expected-identity-record-sha256",
            qualified.qualification_sha256,
        )
    )
    command = render_governed_srun(image, mounts=mount_text, bootstrap_args=tuple(bootstrap_args), legacy=True)
    lines.extend((command, ""))
    return "\n".join(lines)


def _postprocessing_governed_mounts(
    authority: PostprocessingRenderInput,
    *,
    cluster_runspec: PurePosixPath,
    cluster_phase_runspec: PurePosixPath,
    cluster_policy: PurePosixPath,
    cluster_qualification: PurePosixPath,
) -> str:
    """Render only frozen data mounts plus the qualified governed inputs."""
    qualified = authority.runspec.payload.qualified_runtime
    mounts: list[tuple[str, str]] = [
        (qualified.source_package_path, f"{LEGACY_SOURCE_PACKAGE_CONTAINER_PATH}:ro"),
        (str(cluster_runspec), f"{LEGACY_RUNSPEC_CONTAINER_PATH}:ro"),
        (str(cluster_phase_runspec), f"{_PHASE_LEGACY_RUNSPEC_CONTAINER_PATH}:ro"),
        (str(cluster_policy), f"{_POLICY_CONTAINER_PATH}:ro"),
        (str(cluster_qualification), f"{_QUALIFICATION_CONTAINER_PATH}:ro"),
        (
            qualified.runtime_ipsae_binary_path,
            f"{LEGACY_RUNTIME_IPSAE_BINARY_CONTAINER_PATH}:ro",
        ),
    ]
    if qualified.toolkit_package_path is not None:
        mounts.insert(1, (qualified.toolkit_package_path, f"{LEGACY_TOOLKIT_PACKAGE_CONTAINER_PATH}:ro"))
    reserved_targets = {
        str(LEGACY_SOURCE_PACKAGE_CONTAINER_PATH),
        str(LEGACY_TOOLKIT_PACKAGE_CONTAINER_PATH),
        str(LEGACY_RUNSPEC_CONTAINER_PATH),
        _PHASE_LEGACY_RUNSPEC_CONTAINER_PATH,
        _POLICY_CONTAINER_PATH,
        _QUALIFICATION_CONTAINER_PATH,
        str(LEGACY_RUNTIME_IPSAE_BINARY_CONTAINER_PATH),
    }
    ambient_sources = {
        str(path)
        for path in (
            authority.legacy_runspec.paths.orchestration_repo,
            authority.legacy_runspec.paths.afdb_toolkit_repo,
            authority.legacy_runspec.paths.legacy_repo,
        )
        if path is not None
    }
    for item in authority.legacy_runspec.container.mounts:
        source = str(item.source)
        target = str(item.target)
        if source in ambient_sources:
            continue
        if target in reserved_targets:
            raise ValueError("postprocessing data mount overlaps a governed container target")
        mounts.append((source, target))
    if len(set(mounts)) != len(mounts):
        raise ValueError("postprocessing governed container mounts must be unique")
    return ",".join(f"{source}:{target}" for source, target in mounts)


def _postprocessing_raw_command(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> str:
    if action.step_name == "acceptance-adjudication":
        return ""
    if authority.legacy_runspec.workflow is None:
        raise ValueError("postprocessing action rendering requires the stored legacy workflow")
    step = next((item for item in authority.legacy_runspec.workflow.steps if item.name == action.step_name), None)
    if step is None or not step.run:
        raise ValueError(f"postprocessing action has no active stored legacy step: {action.step_name!r}")
    return resolve_workflow_step_command(authority.legacy_runspec, step, legacy=True)


def _container_action_body(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
    *,
    raw_command: str,
) -> str:
    evidence_dir = authority.runspec.payload.attempt_paths.evidence_dir
    capture_path = f"{evidence_dir}/phase-acceptance/{action.step_name}-capture.json"
    raw_stdout_path = f"{evidence_dir}/phase-acceptance/raw/{action.step_name}.stdout"
    raw_stderr_path = f"{evidence_dir}/phase-acceptance/raw/{action.step_name}.stderr"
    lines = [
        "set -euo pipefail",
        f"AFCDB_RUNSPEC={shlex.quote(str(LEGACY_RUNSPEC_CONTAINER_PATH))}",
        "ORCHESTRATION_ROOT={AFCDB_SOURCE_ROOT}",
        "TOOLKIT_ROOT={AFCDB_TOOLKIT_ROOT}",
        f'PYTHON_BIN="${{AFCDB_ORCHESTRATION_PYTHON:-{LEGACY_PIXI_PYTHON_PATH}}}"',
        (
            'export PYTHONPATH="${TOOLKIT_ROOT}:${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:'
            '${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src"'
        ),
        (
            'afcdb-orchestration-runtime() { "$PYTHON_BIN" -c '
            '"from afcdb.orchestration.runtime.cli import cli; cli()" "$@"; }'
        ),
        "",
    ]
    if action.step_name == "acceptance-adjudication":
        lines.append(
            " ".join(
                (
                    '"$PYTHON_BIN"',
                    "-m",
                    "afcdb.orchestration.runtime.postprocessing.phase_acceptance",
                    "adjudicate",
                    "--policy",
                    shlex.quote(_POLICY_CONTAINER_PATH),
                    "--expected-policy-sha256",
                    shlex.quote(authority.runspec.payload.acceptance_policy.sha256),
                    "--evidence-root",
                    shlex.quote(evidence_dir),
                    "--phase-run-id",
                    shlex.quote(authority.phase_run_id),
                    "--attempt-id",
                    shlex.quote(authority.attempt_id),
                    "--phase-runspec",
                    shlex.quote(_PHASE_LEGACY_RUNSPEC_CONTAINER_PATH),
                    "--expected-phase-runspec-digest",
                    shlex.quote(authority.runspec.digest),
                    "--output",
                    shlex.quote(f"{evidence_dir}/phase-acceptance/adjudication.json"),
                )
            )
        )
        lines.append(
            " ".join(
                (
                    '"$PYTHON_BIN"',
                    "-m",
                    "afcdb.orchestration.runtime.postprocessing.finalization_bundle",
                    "publish-action09",
                    "--phase-runspec",
                    shlex.quote(_PHASE_LEGACY_RUNSPEC_CONTAINER_PATH),
                    "--execution-projection",
                    shlex.quote(str(LEGACY_RUNSPEC_CONTAINER_PATH)),
                    "--acceptance-policy",
                    shlex.quote(_POLICY_CONTAINER_PATH),
                    "--command-digest",
                    shlex.quote(postprocessing_action_command_digest(authority, action)),
                    "--workers",
                    '"${SLURM_CPUS_PER_TASK:?}"',
                )
            )
        )
        return "\n".join(lines)
    if action.step_name in {
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    }:
        lines.extend(
            (
                f"mkdir -p {shlex.quote(evidence_dir + '/phase-acceptance/raw')}",
                "set +e",
                "set -o noclobber",
                f"exec 3> {shlex.quote(raw_stdout_path)} || exit 73",
                f"exec 4> {shlex.quote(raw_stderr_path)} || exit 73",
                "{",
                raw_command,
                "} >&3 2>&4",
                "AFCDB_RAW_EXIT=$?",
                "exec 3>&-",
                "exec 4>&-",
                "set +o noclobber",
                f"cat {shlex.quote(raw_stdout_path)}",
                f"cat {shlex.quote(raw_stderr_path)} >&2",
                "set -e",
            )
        )
        lines.append(
            " ".join(
                (
                    '"$PYTHON_BIN"',
                    "-m",
                    "afcdb.orchestration.runtime.postprocessing.phase_acceptance",
                    "capture",
                    "--policy",
                    shlex.quote(_POLICY_CONTAINER_PATH),
                    "--expected-policy-sha256",
                    shlex.quote(authority.runspec.payload.acceptance_policy.sha256),
                    "--evidence-root",
                    shlex.quote(evidence_dir),
                    "--phase-run-id",
                    shlex.quote(authority.phase_run_id),
                    "--attempt-id",
                    shlex.quote(authority.attempt_id),
                    "--action-id",
                    shlex.quote(action.action_id),
                    "--step-name",
                    shlex.quote(action.step_name),
                    "--raw-exit-code",
                    '"$AFCDB_RAW_EXIT"',
                    "--raw-stdout",
                    shlex.quote(raw_stdout_path),
                    "--raw-stderr",
                    shlex.quote(raw_stderr_path),
                    "--output",
                    shlex.quote(capture_path),
                )
            )
        )
    else:
        if action.step_name == "preflight":
            lines.append(
                " ".join(
                    (
                        '"$PYTHON_BIN"',
                        "-m",
                        "afcdb.orchestration.runtime.postprocessing.phase_artifacts",
                        "attest-inputs",
                        "--phase-runspec",
                        shlex.quote(_PHASE_LEGACY_RUNSPEC_CONTAINER_PATH),
                        "--runtime-qualification",
                        shlex.quote(_QUALIFICATION_CONTAINER_PATH),
                        "--evidence-root",
                        shlex.quote(evidence_dir),
                        "--output",
                        shlex.quote(f"{evidence_dir}/phase-inputs/runtime-input-attestations.json"),
                    )
                )
            )
        lines.append(raw_command)
        if action.step_name == "analysis-finalize":
            lines.append(
                " ".join(
                    (
                        '"$PYTHON_BIN"',
                        "-m",
                        "afcdb.orchestration.runtime.postprocessing.finalization_bundle",
                        "scientific-root",
                        "--phase-runspec",
                        shlex.quote(_PHASE_LEGACY_RUNSPEC_CONTAINER_PATH),
                        "--workers",
                        '"${SLURM_CPUS_PER_TASK:?}"',
                    )
                )
            )
    lines.append(_runtime_action_success_command(authority, action))
    return "\n".join(lines)


def postprocessing_action_command_digest(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> str:
    """Bind Runtime success evidence to one exact stored action command."""
    return canonical_mapping_digest(
        {
            "schema_version": 1,
            "command_kind": "postprocessing-phase-action-v1",
            "phase_runspec_digest": authority.runspec.digest,
            "action_id": action.action_id,
            "runtime_action_digest": action.digest,
            "raw_command": _postprocessing_raw_command(authority, action),
        }
    )


def _runtime_action_success_command(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> str:
    parts = [
        '"$PYTHON_BIN"',
        "-m",
        "afcdb.orchestration.runtime.postprocessing.finalization_bundle",
        "record-action",
        "--phase-runspec",
        shlex.quote(_PHASE_LEGACY_RUNSPEC_CONTAINER_PATH),
        "--action-id",
        shlex.quote(action.action_id),
        "--command-digest",
        shlex.quote(postprocessing_action_command_digest(authority, action)),
        "--scheduler-job-id",
    ]
    if action.expected_task_indexes:
        parts.append('"${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID:?}"')
        parts.extend(("--task-index", '"${SLURM_ARRAY_TASK_ID:?}"'))
    else:
        parts.append('"${SLURM_JOB_ID:?}"')
    return " ".join(parts)


def _cluster_attempt_root(runspec: ReadablePostprocessingPhaseRunSpec, *, legacy: bool = False) -> PurePosixPath:
    phase_runs_dir = "afcdb-phase-runs" if legacy else "bspp-phase-runs"
    return PurePosixPath(runspec.cluster.staging_root) / phase_runs_dir / runspec.phase_run_id / runspec.attempt_id


__all__ = [
    "HISTORICAL_POSTPROCESSING_RENDERERS",
    "HistoricalPostprocessingRenderer",
    "historical_postprocessing_renderer",
    "render_historical_postprocessing_action_script_v1",
]
