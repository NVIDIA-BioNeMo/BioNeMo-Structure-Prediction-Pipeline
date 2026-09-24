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

"""Renderer contract 3 for explicit postprocessing V3 authorities."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingRuntimeAction,
)
from bspp.orchestration.contract.postprocessing_autorequeue_contract import (
    AUTOREQUEUE_ACTION_ID_ENV,
    AUTOREQUEUE_COMMAND_DIGEST_ENV,
    AUTOREQUEUE_ENV_NAMES,
    AUTOREQUEUE_PHASE_RUNSPEC_ENV,
    AUTOREQUEUE_RESTART_COUNT_FILE_ENV,
    AUTOREQUEUE_TASK_INDEX_ENV,
    POSTPROCESSING_RESTART_COUNT_FILENAME,
    postprocessing_restart_evidence_relative_dir,
)
from bspp.orchestration.contract.postprocessing_execution import PostprocessingCredentialMountSnapshot
from bspp.orchestration.contract.postprocessing_runspec import (
    ReadablePostprocessingPhaseRunSpec,
)
from bspp.orchestration.contract.postprocessing_runspec_v3 import (
    PostprocessingPhaseRunSpecV3,
)
from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.control.execution_bootstrap import (
    ImageIdentity,
    render_governed_srun,
    slurm_environment_contract,
)
from bspp.orchestration.control.postprocessing_action_commands import resolve_postprocessing_action_command
from bspp.orchestration.control.postprocessing_identity_v3 import resolve_action_semantic_v3
from bspp.orchestration.control.postprocessing_scheduler_identity import (
    postprocessing_scheduler_correlation_token,
)
from bspp.orchestration.control.workflow_rendering import (
    PIXI_PYTHON_PATH,
    RUNSPEC_CONTAINER_PATH,
    RUNTIME_IPSAE_BINARY_CONTAINER_PATH,
    SOURCE_PACKAGE_CONTAINER_PATH,
    TOOLKIT_PACKAGE_CONTAINER_PATH,
)

_POLICY_CONTAINER_PATH = "/run/bspp/postprocessing-acceptance-policy.json"
_QUALIFICATION_CONTAINER_PATH = "/run/bspp/runtime-qualification.json"
_PHASE_RUNSPEC_CONTAINER_PATH = "/run/bspp/postprocessing-phase-runspec.json"
_AWS_SHARED_CREDENTIALS_CONTAINER_ROOT = Path("/workspace/bspp-aws")


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


def render_postprocessing_action_script_v3_contract3(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> str:
    """Render a V3 action with exact scalar/array Slurm log identities."""
    if not isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
        raise ValueError("renderer contract 3 requires a postprocessing V3 RunSpec")
    aws_profile_name = _aws_profile_name(authority)
    credential_mounts = PostprocessingCredentialMountSnapshot(
        aws_shared_credentials_file=str(Path.home() / ".aws" / "credentials") if aws_profile_name else None,
        aws_config_file=str(Path.home() / ".aws" / "config") if aws_profile_name else None,
    )
    return _render_postprocessing_action_script_with_credential_mounts(authority, action, credential_mounts)


def _render_postprocessing_action_script_with_credential_mounts(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
    credential_mounts: PostprocessingCredentialMountSnapshot,
    *,
    slurm_environment_contract_version: int = 1,
) -> str:
    """Shared V3 rendering body with an explicit credential-locator input."""
    runspec = authority.runspec
    if not isinstance(runspec, PostprocessingPhaseRunSpecV3):
        raise ValueError("postprocessing V3 renderer requires a postprocessing V3 RunSpec")
    stored_semantic = next(
        (item for item in runspec.payload.action_semantics.actions if item.action_id == action.action_id),
        None,
    )
    resolved_semantic = resolve_action_semantic_v3(
        action,
        legacy_runspec=authority.legacy_runspec,
        normalized_arguments=runspec.payload.action_semantics.semantic_fields,
    )
    if stored_semantic != resolved_semantic:
        raise ValueError(f"stored action semantics differ from rendered command for {action.action_id!r}")
    return _render_postprocessing_action_script_body(
        authority,
        action,
        credential_mounts=credential_mounts,
        slurm_environment_contract_version=slurm_environment_contract_version,
    )


def _render_postprocessing_action_script_body(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
    *,
    credential_mounts: PostprocessingCredentialMountSnapshot,
    slurm_environment_contract_version: int,
) -> str:
    """Render the frozen renderer-contract-v1 body after family validation."""
    runspec = authority.runspec
    if not isinstance(runspec, PostprocessingPhaseRunSpecV3):
        raise ValueError("postprocessing V3 renderer body requires a postprocessing V3 RunSpec")
    cluster_attempt = _cluster_attempt_root(authority.runspec)
    cluster_runspec = cluster_attempt / "legacy-runspec.yaml"
    cluster_phase_runspec = cluster_attempt / "phase-runspec.json"
    cluster_policy = cluster_attempt / "acceptance-policy.json"
    cluster_qualification = cluster_attempt / "runtime-qualification.json"
    evidence_dir = authority.runspec.payload.attempt_paths.evidence_dir
    scheduler_token = postprocessing_scheduler_correlation_token(authority.runspec, action)
    log_job_token = "%A_%a" if action.expected_task_indexes else "%j"
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
        f"#SBATCH --output={evidence_dir}/slurm-logs/{action.step_name}.{log_job_token}.out",
        f"#SBATCH --error={evidence_dir}/slurm-logs/{action.step_name}.{log_job_token}.err",
    ]
    if action.resources.gres:
        lines.append(f"#SBATCH --gres={action.resources.gres}")
    if action.normalized_array is not None:
        lines.append(f"#SBATCH --array={action.normalized_array}")
    if action.resources.nodelist is not None:
        lines.append(f"#SBATCH --nodelist={action.resources.nodelist}")
    autorequeue_enabled = _autorequeue_enabled_for_action(runspec, action)
    if autorequeue_enabled:
        lines.append("#SBATCH --requeue")
    lines.extend(["", "set -euo pipefail", f"mkdir -p {shlex.quote(evidence_dir + '/slurm-logs')}", ""])
    if autorequeue_enabled:
        lines.extend(_autorequeue_outer_shell_capture(evidence_dir, action))
    raw_command = _postprocessing_raw_command(authority, action)
    container_body = _container_action_body(
        authority,
        action,
        raw_command=raw_command,
        slurm_environment_contract_version=slurm_environment_contract_version,
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
        credential_mounts=credential_mounts,
    )
    bootstrap_args = [
        "run",
        "--identity-record",
        _QUALIFICATION_CONTAINER_PATH,
        "--source-package",
        str(SOURCE_PACKAGE_CONTAINER_PATH),
    ]
    if qualified.source_kind == "override":
        bootstrap_args.extend(("--toolkit-package", str(TOOLKIT_PACKAGE_CONTAINER_PATH)))
    bootstrap_args.extend(
        (
            "--runtime-ipsae-binary",
            str(RUNTIME_IPSAE_BINARY_CONTAINER_PATH),
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
    command = render_governed_srun(
        image,
        mounts=mount_text,
        bootstrap_args=tuple(bootstrap_args),
        slurm_environment_contract_version=slurm_environment_contract_version,
    )
    lines.extend((command, ""))
    return "\n".join(lines)


def _postprocessing_governed_mounts(
    authority: PostprocessingRenderInput,
    *,
    cluster_runspec: PurePosixPath,
    cluster_phase_runspec: PurePosixPath,
    cluster_policy: PurePosixPath,
    cluster_qualification: PurePosixPath,
    credential_mounts: PostprocessingCredentialMountSnapshot,
) -> str:
    """Render only frozen data mounts plus the qualified governed inputs."""
    qualified = authority.runspec.payload.qualified_runtime
    mounts: list[tuple[str, str]] = [
        (qualified.source_package_path, f"{SOURCE_PACKAGE_CONTAINER_PATH}:ro"),
        (str(cluster_runspec), f"{RUNSPEC_CONTAINER_PATH}:ro"),
        (str(cluster_phase_runspec), f"{_PHASE_RUNSPEC_CONTAINER_PATH}:ro"),
        (str(cluster_policy), f"{_POLICY_CONTAINER_PATH}:ro"),
        (str(cluster_qualification), f"{_QUALIFICATION_CONTAINER_PATH}:ro"),
        (
            qualified.runtime_ipsae_binary_path,
            f"{RUNTIME_IPSAE_BINARY_CONTAINER_PATH}:ro",
        ),
    ]
    if qualified.toolkit_package_path is not None:
        mounts.insert(1, (qualified.toolkit_package_path, f"{TOOLKIT_PACKAGE_CONTAINER_PATH}:ro"))
    if _aws_profile_name(authority) is not None:
        credentials_source = credential_mounts.aws_shared_credentials_file
        config_source = credential_mounts.aws_config_file
        if credentials_source is None or config_source is None:
            raise ValueError("postprocessing AWS renderer requires both frozen credential mount locators")
        mounts.extend(
            (
                (
                    credentials_source,
                    f"{_AWS_SHARED_CREDENTIALS_CONTAINER_ROOT / 'credentials'}:ro",
                ),
                (
                    config_source,
                    f"{_AWS_SHARED_CREDENTIALS_CONTAINER_ROOT / 'config'}:ro",
                ),
            )
        )
    reserved_targets = {
        str(SOURCE_PACKAGE_CONTAINER_PATH),
        str(TOOLKIT_PACKAGE_CONTAINER_PATH),
        str(RUNSPEC_CONTAINER_PATH),
        _PHASE_RUNSPEC_CONTAINER_PATH,
        _POLICY_CONTAINER_PATH,
        _QUALIFICATION_CONTAINER_PATH,
        str(RUNTIME_IPSAE_BINARY_CONTAINER_PATH),
        str(_AWS_SHARED_CREDENTIALS_CONTAINER_ROOT),
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
        target = _canonical_container_target(item.target)
        if source in ambient_sources:
            continue
        if any(_container_targets_overlap(target, reserved) for reserved in reserved_targets):
            raise ValueError("postprocessing data mount overlaps a governed container target")
        mounts.append((source, str(target)))
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
    return resolve_postprocessing_action_command(authority.legacy_runspec, action)


def _container_action_body(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
    *,
    raw_command: str,
    slurm_environment_contract_version: int,
) -> str:
    runspec = authority.runspec
    if not isinstance(runspec, PostprocessingPhaseRunSpecV3):
        raise ValueError("postprocessing V3 renderer body requires a postprocessing V3 RunSpec")
    evidence_dir = runspec.payload.attempt_paths.evidence_dir
    autorequeue_enabled = _autorequeue_enabled_for_action(runspec, action)
    capture_path = f"{evidence_dir}/phase-acceptance/{action.step_name}-capture.json"
    raw_stdout_path = f"{evidence_dir}/phase-acceptance/raw/{action.step_name}.stdout"
    raw_stderr_path = f"{evidence_dir}/phase-acceptance/raw/{action.step_name}.stderr"
    if autorequeue_enabled:
        runtime_dispatch = (
            'bspp-orchestration-runtime() { "$PYTHON_BIN" -c '
            '"from bspp.orchestration.runtime.cli import main; main()" "$@"; }'
        )
    else:
        runtime_dispatch = (
            'bspp-orchestration-runtime() { "$PYTHON_BIN" -c '
            '"from bspp.orchestration.runtime.cli import cli; cli()" "$@"; }'
        )
    lines = [
        "set -euo pipefail",
        f"BSPP_RUNSPEC={shlex.quote(str(RUNSPEC_CONTAINER_PATH))}",
        "ORCHESTRATION_ROOT={BSPP_SOURCE_ROOT}",
        "TOOLKIT_ROOT={BSPP_TOOLKIT_ROOT}",
        f'PYTHON_BIN="${{BSPP_ORCHESTRATION_PYTHON:-{PIXI_PYTHON_PATH}}}"',
        (
            'export PYTHONPATH="${TOOLKIT_ROOT}:${ORCHESTRATION_ROOT}/packages/orchestration-contract/src:'
            '${ORCHESTRATION_ROOT}/packages/orchestration-runtime/src"'
        ),
        runtime_dispatch,
    ]
    if autorequeue_enabled:
        lines.extend(_autorequeue_context_export(authority, action))
    lines.append("")
    aws_profile = _aws_profile_name(authority)
    if aws_profile is not None:
        credentials_path = _AWS_SHARED_CREDENTIALS_CONTAINER_ROOT / "credentials"
        config_path = _AWS_SHARED_CREDENTIALS_CONTAINER_ROOT / "config"
        lines.extend(
            (
                f"AWS_SHARED_CREDENTIALS_FILE={shlex.quote(str(credentials_path))}",
                f"AWS_CONFIG_FILE={shlex.quote(str(config_path))}",
                "export AWS_SHARED_CREDENTIALS_FILE AWS_CONFIG_FILE",
            )
        )
        if aws_profile not in {"auto", "profile:auto"}:
            lines.extend((f"AWS_PROFILE={shlex.quote(aws_profile.removeprefix('profile:'))}", "export AWS_PROFILE"))
    if action.step_name == "acceptance-adjudication":
        lines.append(
            " ".join(
                (
                    '"$PYTHON_BIN"',
                    "-m",
                    "bspp.orchestration.runtime.postprocessing.phase_acceptance",
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
                    shlex.quote(_PHASE_RUNSPEC_CONTAINER_PATH),
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
                    "bspp.orchestration.runtime.postprocessing.finalization_bundle",
                    "publish-action09",
                    "--phase-runspec",
                    shlex.quote(_PHASE_RUNSPEC_CONTAINER_PATH),
                    "--execution-projection",
                    shlex.quote(str(RUNSPEC_CONTAINER_PATH)),
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
                "BSPP_RAW_EXIT=$?",
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
                    "bspp.orchestration.runtime.postprocessing.phase_acceptance",
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
                    '"$BSPP_RAW_EXIT"',
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
                        "bspp.orchestration.runtime.postprocessing.phase_artifacts",
                        "attest-inputs",
                        "--phase-runspec",
                        shlex.quote(_PHASE_RUNSPEC_CONTAINER_PATH),
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
                        "bspp.orchestration.runtime.postprocessing.finalization_bundle",
                        "scientific-root",
                        "--phase-runspec",
                        shlex.quote(_PHASE_RUNSPEC_CONTAINER_PATH),
                        "--workers",
                        '"${SLURM_CPUS_PER_TASK:?}"',
                    )
                )
            )
    lines.append(
        _runtime_action_success_command(
            authority,
            action,
            slurm_environment_contract_version=slurm_environment_contract_version,
        )
    )
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
    *,
    slurm_environment_contract_version: int,
) -> str:
    parts = [
        '"$PYTHON_BIN"',
        "-m",
        "bspp.orchestration.runtime.postprocessing.finalization_bundle",
        "record-action",
        "--phase-runspec",
        shlex.quote(_PHASE_RUNSPEC_CONTAINER_PATH),
        "--action-id",
        shlex.quote(action.action_id),
        "--command-digest",
        shlex.quote(postprocessing_action_command_digest(authority, action)),
        "--scheduler-job-id",
    ]
    if action.expected_task_indexes:
        parts.append(slurm_environment_contract(slurm_environment_contract_version).array_success_job_id)
        parts.extend(("--task-index", '"${SLURM_ARRAY_TASK_ID:?}"'))
    else:
        parts.append('"${SLURM_JOB_ID:?}"')
    return " ".join(parts)


def _cluster_attempt_root(runspec: ReadablePostprocessingPhaseRunSpec) -> PurePosixPath:
    return PurePosixPath(runspec.cluster.staging_root) / "bspp-phase-runs" / runspec.phase_run_id / runspec.attempt_id


def _aws_profile_name(authority: PostprocessingRenderInput) -> str | None:
    ref = authority.legacy_runspec.secrets.s3_credentials_ref
    return ref.target if ref.scheme == "aws" else None


def _canonical_container_target(target: Path) -> PurePosixPath:
    path = PurePosixPath(target)
    if not path.is_absolute():
        raise ValueError("postprocessing data mount target must be absolute")
    if path.anchor != "/":
        raise ValueError("postprocessing data mount target must use a single-root absolute path")
    if ".." in path.parts:
        raise ValueError("postprocessing data mount target must not contain traversal aliases")
    if str(path) != str(target):
        raise ValueError("postprocessing data mount target must be canonically lexical")
    return path


def _container_targets_overlap(left: PurePosixPath, right: str) -> bool:
    right_path = PurePosixPath(right)
    return left == right_path or left in right_path.parents or right_path in left.parents


def _autorequeue_enabled_for_action(
    runspec: PostprocessingPhaseRunSpecV3,
    action: PostprocessingRuntimeAction,
) -> bool:
    policy = runspec.payload.autorequeue_policy
    return policy.mode == "enabled" and action.action_id in policy.action_ids


def _autorequeue_restart_dir(evidence_dir: str, action: PostprocessingRuntimeAction) -> str:
    return f"{evidence_dir}/{postprocessing_restart_evidence_relative_dir(action.action_id)}"


def _autorequeue_restart_count_file_shell(evidence_dir: str, action: PostprocessingRuntimeAction) -> str:
    base = f"{_autorequeue_restart_dir(evidence_dir, action)}/{POSTPROCESSING_RESTART_COUNT_FILENAME}"
    if action.expected_task_indexes:
        return f"{shlex.quote(base)}${{SLURM_ARRAY_TASK_ID}}"
    return shlex.quote(base)


def _autorequeue_outer_shell_capture(evidence_dir: str, action: PostprocessingRuntimeAction) -> list[str]:
    restart_dir = _autorequeue_restart_dir(evidence_dir, action)
    restart_count_file = _autorequeue_restart_count_file_shell(evidence_dir, action)
    return [
        f"mkdir -p {shlex.quote(restart_dir)}",
        f"printf '%s\\n' \"${{SLURM_RESTART_COUNT:-0}}\" > {restart_count_file}",
        "",
    ]


def _autorequeue_context_export(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> list[str]:
    evidence_dir = authority.runspec.payload.attempt_paths.evidence_dir
    command_digest = postprocessing_action_command_digest(authority, action)
    restart_count_file = _autorequeue_restart_count_file_shell(evidence_dir, action)
    lines = [
        f"{AUTOREQUEUE_PHASE_RUNSPEC_ENV}={shlex.quote(_PHASE_RUNSPEC_CONTAINER_PATH)}",
        f"{AUTOREQUEUE_ACTION_ID_ENV}={shlex.quote(action.action_id)}",
        f"{AUTOREQUEUE_COMMAND_DIGEST_ENV}={shlex.quote(command_digest)}",
        f"{AUTOREQUEUE_RESTART_COUNT_FILE_ENV}={restart_count_file}",
    ]
    export_names = " ".join(AUTOREQUEUE_ENV_NAMES[:4])
    if action.expected_task_indexes:
        lines.append(f'{AUTOREQUEUE_TASK_INDEX_ENV}="${{SLURM_ARRAY_TASK_ID:?}}"')
        export_names += f" {AUTOREQUEUE_TASK_INDEX_ENV}"
    lines.append(f"export {export_names}")
    return lines


__all__ = [
    "PostprocessingRenderInput",
    "postprocessing_action_command_digest",
    "render_postprocessing_action_script_v3_contract3",
]
