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

"""Pure direct-Slurm rendering for an immutable preprocessing Phase RunSpec."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shlex
from pathlib import PurePosixPath
from typing import cast

from bspp.orchestration.contract.database_placement import (
    DatabaseAccessPolicy,
    DatabasePlacementBranch,
    DatabaseProfileStagingSnapshot,
)
from bspp.orchestration.contract.phase import PhaseRunSpec, canonical_mapping_digest
from bspp.orchestration.contract.phase_carry_forward import AttemptCarryForwardRecord
from bspp.orchestration.contract.phase_submission import (
    PhaseActionSubmissionPlan,
    PhaseContainerMountDescriptor,
    PhaseMountOrigin,
    PhaseSubmissionIntendedPayload,
    phase_action_scheduler_correlation_token,
    phase_action_submission_identity_mapping,
    phase_submission_id,
)
from bspp.orchestration.contract.preprocessing_runtime import (
    PREPROCESSING_RUNTIME_COMMAND,
    preprocessing_runtime_tuple_id,
)
from bspp.orchestration.control.preprocessing_runtime_qualification import derive_smoke_gres

_FINALIZE_COMMAND = ("bspp-orchestration-runtime", "preprocessing", "finalize-chunk")
_PLACE_DATABASE_COMMAND = ("bspp-orchestration-runtime", "preprocessing", "place-database")
_SELECT_DATABASE_SCIENCE_BRANCH_COMMAND = (
    "bspp-orchestration-runtime",
    "preprocessing",
    "select-database-science-branch",
)
_STAGE_INPUT_COMMAND = (
    "bspp-orchestration-runtime",
    "preprocessing",
    "stage-input",
)


def render_phase_submission_intent(
    runspec: PhaseRunSpec,
    *,
    phase_runspec_location: str,
    phase_runspec_document_sha256: str,
    carry_forward_record: AttemptCarryForwardRecord | None = None,
    carry_forward_document_sha256: str | None = None,
) -> PhaseSubmissionIntendedPayload:
    """Render every declared action without filesystem, clock, or scheduler access."""
    if re.fullmatch(r"[0-9a-f]{64}", phase_runspec_document_sha256) is None:
        raise ValueError("Phase RunSpec document SHA-256 must be lowercase SHA-256")
    cluster_runspec_path = str(
        PurePosixPath(runspec.cluster.staging_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / "phase-runspec.json"
    )
    if (carry_forward_record is None) != (carry_forward_document_sha256 is None):
        raise ValueError("carry-forward record and document SHA-256 must be present together")
    if carry_forward_record is not None:
        reference = runspec.carry_forward
        if (
            reference is None
            or reference.attempt_carry_forward_id != carry_forward_record.attempt_carry_forward_id
            or reference.digest != carry_forward_record.digest
            or carry_forward_record.phase_run_id != runspec.phase_run_id
            or carry_forward_record.target_attempt_id != runspec.attempt_id
            or carry_forward_record.workspace.target_action_id != runspec.payload.actions[0].action_id
        ):
            raise ValueError("carried rendering requires the exact RunSpec-referenced carry record")
    elif runspec.carry_forward is not None:
        raise ValueError("carried RunSpec rendering requires its complete carry record")
    cluster_carry_path = (
        str(
            PurePosixPath(runspec.cluster.staging_root)
            / "bspp-phase-runs"
            / runspec.phase_run_id
            / runspec.attempt_id
            / "attempt-carry-forward.json"
        )
        if carry_forward_record is not None
        else None
    )
    identities: list[dict[str, object]] = []
    carry_mounts_by_action: list[tuple[PhaseContainerMountDescriptor, ...]] = []
    for action in runspec.payload.actions:
        action_root = (
            PurePosixPath(runspec.cluster.output_root)
            / "bspp-phase-runs"
            / runspec.phase_run_id
            / runspec.attempt_id
            / action.action_id
        )
        cluster_script_path = str(
            PurePosixPath(runspec.cluster.staging_root)
            / "bspp-phase-runs"
            / runspec.phase_run_id
            / runspec.attempt_id
            / "actions"
            / f"{action.action_id}.sbatch"
        )
        carry_mounts = (
            _carry_mount_descriptors(carry_forward_record, cluster_carry_path)
            if carry_forward_record is not None and cluster_carry_path is not None
            else ()
        )
        carry_mounts_by_action.append(carry_mounts)
        identities.append(
            phase_action_submission_identity_mapping(
                action_id=action.action_id,
                runtime_action_digest=canonical_mapping_digest(action.to_mapping()),
                dependency_action_ids=action.dependencies,
                cluster_script_path=cluster_script_path,
                action_evidence_path=str(action_root / "action-evidence.json"),
                handoff_path=str(action_root / "handoff"),
                carry_forward_record_path=cluster_carry_path,
                carry_forward_record_sha256=carry_forward_document_sha256,
                carry_forward_mounts=carry_mounts,
            )
        )
    qualification_tuple = runspec.cluster.preprocessing_runtime.qualification_tuple
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    submission_id = phase_submission_id(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        phase_runspec_document_sha256=phase_runspec_document_sha256,
        qualification_tuple_id=tuple_id,
        actions=identities,
    )
    plans: list[PhaseActionSubmissionPlan] = []
    for action, identity, carry_mounts in zip(
        runspec.payload.actions,
        identities,
        carry_mounts_by_action,
        strict=True,
    ):
        correlation = phase_action_scheduler_correlation_token(submission_id, action.action_id)
        job_name = correlation
        script = _render_action_script(
            runspec,
            action_id=action.action_id,
            cluster_runspec_path=cluster_runspec_path,
            runspec_document_sha256=phase_runspec_document_sha256,
            action_evidence_path=str(identity["action_evidence_path"]),
            handoff_path=str(identity["handoff_path"]),
            job_name=job_name,
            correlation=correlation,
            carry_forward_record=carry_forward_record,
            carry_forward_record_path=cluster_carry_path,
            carry_forward_mounts=carry_mounts,
            submission_id=submission_id,
        )
        plans.append(
            PhaseActionSubmissionPlan(
                action_id=action.action_id,
                runtime_action_digest=str(identity["runtime_action_digest"]),
                dependency_action_ids=action.dependencies,
                cluster_script_path=str(identity["cluster_script_path"]),
                script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                script_body=script,
                job_name=job_name,
                scheduler_correlation_token=correlation,
                action_evidence_path=str(identity["action_evidence_path"]),
                handoff_path=str(identity["handoff_path"]),
                carry_forward_record_path=cluster_carry_path,
                carry_forward_record_sha256=carry_forward_document_sha256,
                carry_forward_mounts=carry_mounts,
                carry_forward_submission_id=submission_id if carry_forward_record is not None else None,
            )
        )
    return PhaseSubmissionIntendedPayload(
        submission_id=submission_id,
        phase_runspec_location=phase_runspec_location,
        phase_runspec_digest=runspec.digest,
        phase_runspec_document_sha256=phase_runspec_document_sha256,
        qualification_tuple_id=tuple_id,
        actions=tuple(plans),
    )


def _render_action_script(
    runspec: PhaseRunSpec,
    *,
    action_id: str,
    cluster_runspec_path: str,
    runspec_document_sha256: str,
    action_evidence_path: str,
    handoff_path: str,
    job_name: str,
    correlation: str,
    carry_forward_record: AttemptCarryForwardRecord | None,
    carry_forward_record_path: str | None,
    carry_forward_mounts: tuple[PhaseContainerMountDescriptor, ...],
    submission_id: str,
) -> str:
    action = next(item for item in runspec.payload.actions if item.action_id == action_id)
    resources = action.resources
    if resources.array is not None:
        raise ValueError("preprocessing Phase Submission does not support Slurm arrays")
    qualification = runspec.cluster.preprocessing_runtime.qualification_tuple
    if action.payload.site.container_image != qualification.cluster_image_path:
        raise ValueError("Runtime Action image does not match qualified preprocessing image")
    action_gres = derive_smoke_gres(
        gres=resources.gres,
        nodes=resources.nodes,
        gpus_per_task=resources.gpus_per_task,
        profile_name=qualification.cluster_profile,
    )
    if action_gres != qualification.gpu_worker_gres:
        raise ValueError("Runtime Action GRES does not match qualified preprocessing resources")
    if resources.nodelist != qualification.nodelist:
        raise ValueError("Runtime Action nodelist does not match qualified preprocessing resources")
    database = runspec.payload.database
    preferred = database.requested_policy == DatabaseAccessPolicy.STAGE_PREFERRED
    execution_lines: tuple[str, ...]
    if preferred:
        if tuple(branch.branch_kind for branch in database.branches) != (
            "staged",
            "direct-capacity-fallback",
        ):
            raise ValueError("stage-preferred rendering requires the exact ordered branch closure")
        staged_branch, fallback_branch = database.branches
        if (
            staged_branch.gpuserver_argv != fallback_branch.gpuserver_argv
            or staged_branch.search_argv != fallback_branch.search_argv
        ):
            raise ValueError("stage-preferred rendering requires identical branch commands")
        branch = staged_branch
    else:
        if (
            database.requested_policy
            not in {
                DatabaseAccessPolicy.DIRECT,
                DatabaseAccessPolicy.STAGE_REQUIRED,
            }
            or len(database.branches) != 1
        ):
            raise ValueError("Runtime Action rendering supports one canonical Database Access closure")
        branch = database.branches[0]
        expected_branch = "direct-requested" if database.requested_policy == DatabaseAccessPolicy.DIRECT else "staged"
        if branch.branch_kind != expected_branch:
            raise ValueError("Runtime Action rendering branch does not match requested Database Access Policy")
        staged_branch = branch
        fallback_branch = branch

    action_root = str(PurePosixPath(action_evidence_path).parent)
    cluster_source_manifest_path = str(PurePosixPath(cluster_runspec_path).parent / "database-source-manifest.json")
    required_mounts = (
        PhaseContainerMountDescriptor(
            source=cluster_runspec_path,
            target=cluster_runspec_path,
            source_kind="file",
            read_only=False,
            origin="runspec",
        ),
        PhaseContainerMountDescriptor(
            source=action_root,
            target=action_root,
            source_kind="directory",
            read_only=False,
            origin="action-root",
        ),
        PhaseContainerMountDescriptor(
            source=str(PurePosixPath(qualification.source_bundle_path).parent),
            target=str(PurePosixPath(qualification.source_bundle_path).parent),
            source_kind="directory",
            read_only=False,
            origin="source-bundle",
        ),
    )
    required_protected_owners = (
        (cluster_runspec_path, required_mounts[0]),
        (action_root, required_mounts[1]),
        (qualification.source_bundle_path, required_mounts[2]),
    )
    scientific_protected_owners = (
        *required_protected_owners,
        *((mount.target, mount) for mount in carry_forward_mounts),
        *(
            ("/run/bspp-carry", mount)
            for mount in carry_forward_mounts
            if PurePosixPath("/run/bspp-carry") in PurePosixPath(mount.target).parents
        ),
    )
    placement_database_mounts = (
        PhaseContainerMountDescriptor(
            source=cluster_source_manifest_path,
            target=cluster_source_manifest_path,
            source_kind="file",
            read_only=True,
            origin="database-protected",
        ),
        *(
            PhaseContainerMountDescriptor(
                source=mount.source,
                target=mount.target,
                source_kind="directory",
                read_only=mount.read_only,
                origin="database-protected",
            )
            for mount in branch.placement_mounts
        ),
    )

    def science_database_mounts(selected_branch: DatabasePlacementBranch) -> tuple[PhaseContainerMountDescriptor, ...]:
        return tuple(
            PhaseContainerMountDescriptor(
                source=mount.source,
                target=mount.target,
                source_kind=(
                    "file" if (mount.source_kind == "file" or mount.purpose == "replica-lease") else "directory"
                ),
                read_only=mount.read_only,
                origin="database-protected",
            )
            for mount in selected_branch.scientific_mounts
        )

    scientific_database_mounts = science_database_mounts(branch)
    staged_scientific_database_mounts = science_database_mounts(staged_branch)
    fallback_scientific_database_mounts = science_database_mounts(fallback_branch)
    if any(candidate.finalization_mounts for candidate in database.branches):
        raise ValueError("Runtime Action finalization must not receive database or lease mounts")
    placement_mounts = _container_mounts(
        declared=(),
        extra=(),
        required=tuple((mount.source, mount.target) for mount in required_mounts),
        database=placement_database_mounts,
        protected=("/usr/local/bin",),
        identity_protected=database.identity_protected_targets,
        protected_owners=(
            *required_protected_owners,
            *((mount.target, mount) for mount in placement_database_mounts),
        ),
    )
    scientific_mounts = _container_mounts(
        declared=action.payload.site.container_mounts,
        extra=tuple((mount.source, mount.target) for mount in runspec.cluster.extra_mounts),
        required=tuple((mount.source, mount.target) for mount in required_mounts),
        carry=carry_forward_mounts,
        protected=(
            "/usr/local/bin",
            action.payload.site.mmseqs_executable,
            action.payload.site.colabfold_search_executable,
            action.payload.site.tar_executable,
            action.payload.site.lz4_executable,
        ),
        identity_protected=runspec.payload.database.identity_protected_targets,
        protected_owners=(
            *scientific_protected_owners,
            *((mount.target, mount) for mount in scientific_database_mounts),
        ),
        database=scientific_database_mounts,
    )
    staged_scientific_mounts = _container_mounts(
        declared=action.payload.site.container_mounts,
        extra=tuple((mount.source, mount.target) for mount in runspec.cluster.extra_mounts),
        required=tuple((mount.source, mount.target) for mount in required_mounts),
        carry=carry_forward_mounts,
        protected=(
            "/usr/local/bin",
            action.payload.site.mmseqs_executable,
            action.payload.site.colabfold_search_executable,
            action.payload.site.tar_executable,
            action.payload.site.lz4_executable,
        ),
        identity_protected=runspec.payload.database.identity_protected_targets,
        protected_owners=(
            *scientific_protected_owners,
            *((mount.target, mount) for mount in staged_scientific_database_mounts),
        ),
        database=staged_scientific_database_mounts,
    )
    fallback_scientific_mounts = _container_mounts(
        declared=action.payload.site.container_mounts,
        extra=tuple((mount.source, mount.target) for mount in runspec.cluster.extra_mounts),
        required=tuple((mount.source, mount.target) for mount in required_mounts),
        carry=carry_forward_mounts,
        protected=(
            "/usr/local/bin",
            action.payload.site.mmseqs_executable,
            action.payload.site.colabfold_search_executable,
            action.payload.site.tar_executable,
            action.payload.site.lz4_executable,
        ),
        identity_protected=runspec.payload.database.identity_protected_targets,
        protected_owners=(
            *scientific_protected_owners,
            *((mount.target, mount) for mount in fallback_scientific_database_mounts),
        ),
        database=fallback_scientific_database_mounts,
    )
    finalization_mounts = _container_mounts(
        declared=action.payload.site.container_mounts,
        extra=tuple((mount.source, mount.target) for mount in runspec.cluster.extra_mounts),
        required=tuple((mount.source, mount.target) for mount in required_mounts),
        protected=("/usr/local/bin",),
        identity_protected=database.identity_protected_targets,
        protected_owners=required_protected_owners,
    )

    def srun_prefix(mounts: str) -> tuple[str, ...]:
        return (
            "srun",
            f"--container-image={qualification.cluster_image_path}",
            f"--container-mounts={mounts}",
            "--no-container-mount-home",
            "/usr/local/bin/entrypoint.sh",
        )

    placement_result_path = str(PurePosixPath(action_root) / "database-placement-result.json")
    placement_failure_path = str(PurePosixPath(action_root) / "database-placement-failure.json")
    place = (
        *srun_prefix(placement_mounts),
        *_PLACE_DATABASE_COMMAND,
        "--phase-runspec",
        cluster_runspec_path,
        "--action-id",
        action_id,
        "--source-manifest",
        cluster_source_manifest_path,
        "--write-result",
        placement_result_path,
        "--write-failure-evidence",
        placement_failure_path,
    )

    def execute_command(mounts: str) -> tuple[str, ...]:
        return (
            *srun_prefix(mounts),
            *PREPROCESSING_RUNTIME_COMMAND,
            "--phase-runspec",
            cluster_runspec_path,
            "--action-id",
            action_id,
            "--write-evidence",
            action_evidence_path,
            "--database-placement-result",
            placement_result_path,
            "--database-placement-failure",
            placement_failure_path,
            "--placement-process-status",
            "__BSPP_PLACEMENT_PROCESS_STATUS__",
        )

    def execute_shell_command(
        argv: tuple[str, ...],
        *,
        status_variable: str = "_placement_status",
    ) -> str:
        return _shell_command(argv).replace(
            "__BSPP_PLACEMENT_PROCESS_STATUS__",
            f'"${status_variable}"',
        )

    execute = execute_command(scientific_mounts)
    staged_execute = execute_command(staged_scientific_mounts)
    fallback_execute = execute_command(fallback_scientific_mounts)
    placement_failure_execute = execute_command(finalization_mounts)
    if carry_forward_record is not None:
        assert carry_forward_record_path is not None
        carry_args = (
            "--carry-forward-record",
            carry_forward_record_path,
            "--phase-submission-id",
            submission_id,
        )
        execute = (*execute, *carry_args)
        staged_execute = (*staged_execute, *carry_args)
        fallback_execute = (*fallback_execute, *carry_args)
    selector = (
        *srun_prefix(finalization_mounts),
        *_SELECT_DATABASE_SCIENCE_BRANCH_COMMAND,
        "--phase-runspec",
        cluster_runspec_path,
        "--action-id",
        action_id,
        "--database-placement-result",
        placement_result_path,
        "--database-placement-failure",
        placement_failure_path,
    )
    finalize = (
        *srun_prefix(finalization_mounts),
        *_FINALIZE_COMMAND,
        "--phase-runspec",
        cluster_runspec_path,
        "--action-evidence",
        action_evidence_path,
        "--write-handoff",
        handoff_path,
    )
    stage_input = (
        *srun_prefix(finalization_mounts),
        *_STAGE_INPUT_COMMAND,
        "--phase-runspec",
        cluster_runspec_path,
        "--workspace-root",
        str(runspec.cluster.project_root),
    )
    if preferred:

        def case_command(argv: tuple[str, ...]) -> str:
            return "    " + execute_shell_command(argv).replace("\n", "\n    ")

        execution_lines = (
            "set +e",
            f'_database_science_branch="$({_shell_command(selector)})"',
            "_selector_status=$?",
            "set -e",
            'if [[ "$_selector_status" -ne 0 ]]; then',
            "  "
            + execute_shell_command(
                placement_failure_execute,
                status_variable="_selector_status",
            ).replace("\n", "\n  "),
            '  exit "$_selector_status"',
            "fi",
            'case "$_database_science_branch" in',
            "  staged)",
            case_command(staged_execute),
            "    ;;",
            "  direct-capacity-fallback)",
            case_command(fallback_execute),
            "    ;;",
            "  placement-failure)",
            case_command(placement_failure_execute),
            "    ;;",
            "  *)",
            '    echo "unknown Database science branch: $_database_science_branch" >&2',
            "    exit 127",
            "    ;;",
            "esac",
        )
    else:
        execution_lines = (execute_shell_command(execute),)

    def indent_line(value: str) -> str:
        return "  " + value.replace("\n", "\n  ")

    placement_resolution_lines = (
        'if [[ "$_placement_status" -ne 0 '
        '|| ! -f "$DATABASE_PLACEMENT_RESULT" '
        '|| -f "$DATABASE_PLACEMENT_FAILURE" ]]; then',
        indent_line(execute_shell_command(placement_failure_execute)),
        "else",
        *(indent_line(line) for line in execution_lines),
        "fi",
    )
    cache_bootstrap_lines: tuple[str, ...] = ()
    if database.staging is not None:
        cache_bootstrap_lines = _acceptance_cache_bootstrap_lines(
            database.staging,
            cluster_image_path=qualification.cluster_image_path,
            required_mounts=required_mounts,
            required_protected_owners=required_protected_owners,
            identity_protected=database.identity_protected_targets,
            tolerate_failure=preferred,
        )
    log_root = PurePosixPath(action_root)
    lines = [
        "#!/usr/bin/env bash",
        "# BSPP preprocessing Phase Runtime Action",
        f"# Phase Run: {runspec.phase_run_id}",
        f"# Attempt: {runspec.attempt_id}",
        f"# RunSpec digest: {runspec.digest}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --comment={correlation}",
        f"#SBATCH --partition={resources.partition}",
        f"#SBATCH --account={runspec.cluster.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
        f"#SBATCH --mem={resources.memory}",
        f"#SBATCH --time={resources.time}",
        f"#SBATCH --output={log_root / 'slurm-%j.out'}",
        f"#SBATCH --error={log_root / 'slurm-%j.err'}",
    ]
    if action_gres:
        lines.append(f"#SBATCH --gres={action_gres}")
    if resources.nodelist is not None:
        lines.append(f"#SBATCH --nodelist={resources.nodelist}")
    lines.extend(
        (
            "",
            "set -euo pipefail",
            f"IMAGE={shlex.quote(qualification.cluster_image_path)}",
            f"SOURCE_BUNDLE={shlex.quote(qualification.source_bundle_path)}",
            f"PHASE_RUNSPEC={shlex.quote(cluster_runspec_path)}",
            f"EXPECTED_IMAGE_SHA256={qualification.cluster_image_sha256}",
            f"EXPECTED_SOURCE_SHA256={qualification.source_bundle_sha256}",
            f"EXPECTED_RUNSPEC_DOCUMENT_SHA256={runspec_document_sha256}",
            f"DATABASE_PLACEMENT_RESULT={shlex.quote(placement_result_path)}",
            f"DATABASE_PLACEMENT_FAILURE={shlex.quote(placement_failure_path)}",
            *(
                _carry_bootstrap_lines(carry_forward_record, submission_id, action_root)
                if carry_forward_record is not None
                else (f"mkdir -p -- {shlex.quote(action_root)}",)
            ),
            '[[ -f "$IMAGE" ]] || { echo "missing qualified preprocessing image: $IMAGE" >&2; exit 127; }',
            '[[ -f "$SOURCE_BUNDLE" ]] || { echo "missing selected Source Bundle: $SOURCE_BUNDLE" >&2; exit 127; }',
            '[[ -f "$PHASE_RUNSPEC" ]] || { echo "missing staged Phase RunSpec: $PHASE_RUNSPEC" >&2; exit 127; }',
            '_image_sha256="$(sha256sum "$IMAGE" | awk \'{print $1}\')"',
            '_source_sha256="$(sha256sum "$SOURCE_BUNDLE" | awk \'{print $1}\')"',
            '_runspec_sha256="$(sha256sum "$PHASE_RUNSPEC" | awk \'{print $1}\')"',
            '[[ "$_image_sha256" == "$EXPECTED_IMAGE_SHA256" ]] || '
            '{ echo "qualified image SHA-256 mismatch" >&2; exit 127; }',
            '[[ "$_source_sha256" == "$EXPECTED_SOURCE_SHA256" ]] || '
            '{ echo "Source Bundle SHA-256 mismatch" >&2; exit 127; }',
            '[[ "$_runspec_sha256" == "$EXPECTED_RUNSPEC_DOCUMENT_SHA256" ]] || '
            '{ echo "Phase RunSpec document SHA-256 mismatch" >&2; exit 127; }',
            f"export BSPP_PREPROCESSING_SOURCE_BUNDLE={shlex.quote(qualification.source_bundle_path)}",
            *cache_bootstrap_lines,
            _shell_command(stage_input),
            "set +e",
            _shell_command(place),
            "_placement_status=$?",
            "set -e",
            *placement_resolution_lines,
            _shell_command(finalize),
            "",
        )
    )
    return "\n".join(lines)


def _container_mounts(
    *,
    declared: tuple[str, ...],
    extra: tuple[tuple[str, str], ...],
    required: tuple[tuple[str, str], ...],
    carry: tuple[PhaseContainerMountDescriptor, ...] = (),
    database: tuple[PhaseContainerMountDescriptor, ...] = (),
    protected: tuple[str, ...] = (),
    identity_protected: tuple[str, ...] = (),
    protected_owners: tuple[tuple[str, PhaseContainerMountDescriptor], ...] = (),
) -> str:
    descriptors: list[PhaseContainerMountDescriptor] = []
    for value in declared:
        if value.endswith(":ro"):
            raise ValueError("authored Phase mounts cannot encode a :ro mode suffix")
        source, separator, target = value.partition(":")
        descriptors.append(
            PhaseContainerMountDescriptor(
                source=source,
                target=target if separator else source,
                source_kind="directory",
                read_only=False,
                origin="authored",
            )
        )
    descriptors.extend(
        PhaseContainerMountDescriptor(
            source=source,
            target=target,
            source_kind="directory",
            read_only=False,
            origin="cluster-extra",
        )
        for source, target in extra
    )
    required_origins = ("runspec", "action-root", "source-bundle")
    descriptors.extend(
        PhaseContainerMountDescriptor(
            source=source,
            target=target,
            source_kind="file" if origin == "runspec" else "directory",
            read_only=False,
            origin=cast("PhaseMountOrigin", origin),
        )
        for (source, target), origin in zip(required, required_origins, strict=True)
    )
    descriptors.extend(carry)
    descriptors.extend(database)
    ordered = _validate_mount_descriptors(
        tuple(descriptors),
        protected=protected,
        identity_protected=identity_protected,
        protected_owners=protected_owners,
    )
    return ",".join(f"{item.source}:{item.target}{':ro' if item.read_only else ''}" for item in ordered)


def _carry_mount_descriptors(
    record: AttemptCarryForwardRecord,
    cluster_carry_path: str,
) -> tuple[PhaseContainerMountDescriptor, ...]:
    mounts: list[PhaseContainerMountDescriptor] = [
        PhaseContainerMountDescriptor(
            source=cluster_carry_path,
            target=cluster_carry_path,
            source_kind="file",
            read_only=True,
            origin="carry-record",
        ),
        PhaseContainerMountDescriptor(
            source=record.workspace.workspace_root,
            target=record.workspace.private_workspace_mount_path,
            source_kind="directory",
            read_only=False,
            origin="carry-workspace",
        ),
    ]
    mounts.extend(
        PhaseContainerMountDescriptor(
            source=item.physical_root,
            target=item.logical_root,
            source_kind="directory",
            read_only=False,
            origin="carry-workspace",
        )
        for item in record.workspace.roots
    )
    mounts.extend(
        PhaseContainerMountDescriptor(
            source=item.source_physical_path,
            target=item.source_private_mount_path,
            source_kind="file",
            read_only=True,
            origin="carry-source",
        )
        for item in record.content
    )
    return _validate_mount_descriptors(tuple(mounts))


def _validate_mount_descriptors(
    descriptors: tuple[PhaseContainerMountDescriptor, ...],
    *,
    protected: tuple[str, ...] = (),
    identity_protected: tuple[str, ...] = (),
    protected_owners: tuple[tuple[str, PhaseContainerMountDescriptor], ...] = (),
) -> tuple[PhaseContainerMountDescriptor, ...]:
    by_target: dict[str, PhaseContainerMountDescriptor] = {}
    ordered: list[PhaseContainerMountDescriptor] = []
    for descriptor in descriptors:
        if descriptor.origin in {"authored", "cluster-extra"} and descriptor.target.startswith("/run/bspp-carry"):
            raise ValueError("authored mounts cannot shadow the private carry namespace")
        existing = by_target.get(descriptor.target)
        if existing is not None and existing != descriptor:
            target = PurePosixPath(descriptor.target)
            if any(_paths_overlap(target, PurePosixPath(value)) for value in (*protected, *identity_protected)):
                raise ValueError(f"container mount shadows protected runtime target: {descriptor.target}")
            raise ValueError(f"conflicting container mount target: {descriptor.target}")
        if existing is None:
            for prior in ordered:
                prior_path = PurePosixPath(prior.target)
                current_path = PurePosixPath(descriptor.target)
                if (prior.source_kind == "file" and prior_path in current_path.parents) or (
                    descriptor.source_kind == "file" and current_path in prior_path.parents
                ):
                    raise ValueError("file mount target cannot contain another mount target")
                if (
                    prior.source_kind == "directory"
                    and descriptor.source_kind == "directory"
                    and (prior_path in current_path.parents or current_path in prior_path.parents)
                    and not _compatible_directory_overlap(prior, descriptor)
                ):
                    raise ValueError("ambiguous nested directory mount targets")
            by_target[descriptor.target] = descriptor
            ordered.append(descriptor)
    _validate_mount_sources(tuple(ordered))
    _validate_protected_mount_targets(
        tuple(ordered),
        protected,
        identity_protected=identity_protected,
        protected_owners=protected_owners,
    )
    return tuple(ordered)


def _compatible_directory_overlap(
    first: PhaseContainerMountDescriptor,
    second: PhaseContainerMountDescriptor,
) -> bool:
    if first.source == first.target and second.source == second.target:
        return True
    first_path = PurePosixPath(first.target)
    second_path = PurePosixPath(second.target)
    parent, child = (first, second) if first_path in second_path.parents else (second, first)
    return parent.origin in {"authored", "cluster-extra"} and child.origin in {
        "action-root",
        "source-bundle",
        "carry-workspace",
    }


def _validate_mount_sources(descriptors: tuple[PhaseContainerMountDescriptor, ...]) -> None:
    by_source: dict[str, PhaseContainerMountDescriptor] = {}
    for descriptor in descriptors:
        prior = by_source.get(descriptor.source)
        if prior is not None and (
            prior.target != descriptor.target
            or prior.source_kind != descriptor.source_kind
            or prior.read_only != descriptor.read_only
        ):
            if (
                prior.origin == "database-protected"
                and descriptor.origin == "database-protected"
                and prior.read_only
                and descriptor.read_only
                and prior.source_kind == "file"
                and descriptor.source_kind == "file"
            ):
                continue
            raise ValueError(f"incompatible container mount source reuse: {descriptor.source}")
        by_source[descriptor.source] = descriptor


def _validate_protected_mount_targets(
    descriptors: tuple[PhaseContainerMountDescriptor, ...],
    protected: tuple[str, ...],
    *,
    identity_protected: tuple[str, ...] = (),
    protected_owners: tuple[tuple[str, PhaseContainerMountDescriptor], ...] = (),
) -> None:
    protected_paths = tuple(PurePosixPath(value) for value in (*protected, *identity_protected))
    owner_paths = tuple(PurePosixPath(value) for value, _owner in protected_owners)
    if any(
        not path.is_absolute() or ".." in path.parts or str(path) != value
        for path, value in zip(
            (*protected_paths, *owner_paths),
            (*protected, *identity_protected, *(value for value, _owner in protected_owners)),
            strict=True,
        )
    ):
        raise ValueError("protected container targets must be absolute normalized paths")
    declared = set(descriptors)
    if any(owner not in declared for _path, owner in protected_owners):
        raise ValueError("protected mount owner must be one rendered descriptor")
    database_paths = tuple(PurePosixPath(value) for value in identity_protected)
    for descriptor in descriptors:
        if descriptor.origin == "database-protected":
            continue
        for endpoint in (PurePosixPath(descriptor.source), PurePosixPath(descriptor.target)):
            if any(_paths_overlap(endpoint, database_path) for database_path in database_paths):
                if descriptor.origin in {"authored", "cluster-extra"}:
                    raise ValueError("authored or profile mount overlaps protected database namespace")
                raise ValueError("container mount overlaps protected database namespace")
    owners_by_path: dict[PurePosixPath, list[PhaseContainerMountDescriptor]] = {}
    for path, (_value, owner) in zip(owner_paths, protected_owners, strict=True):
        owners_by_path.setdefault(path, []).append(owner)
    protections = (
        *((path, False, ()) for path in (PurePosixPath(value) for value in protected)),
        *((path, True, ()) for path in (PurePosixPath(value) for value in identity_protected)),
        *((path, False, tuple(owners)) for path, owners in owners_by_path.items()),
    )
    for descriptor in descriptors:
        target = PurePosixPath(descriptor.target)
        for protected_path, allow_identity, owners in protections:
            if not _paths_overlap(target, protected_path):
                continue
            if descriptor in owners:
                continue
            if allow_identity and descriptor.origin == "database-protected" and _paths_overlap(target, protected_path):
                continue
            if descriptor.origin in {"authored", "cluster-extra"} and descriptor.source == descriptor.target:
                continue
            if (
                allow_identity
                and descriptor.origin in {"authored", "cluster-extra"}
                and descriptor.source == descriptor.target
                and target == protected_path
            ):
                continue
            if descriptor.origin in {"authored", "cluster-extra"} and any(
                target in PurePosixPath(owner.target).parents
                and (
                    PurePosixPath(owner.target) == protected_path
                    or PurePosixPath(owner.target) in protected_path.parents
                )
                for owner in owners
            ):
                continue
            raise ValueError(f"container mount shadows protected runtime target: {protected_path}")


def _paths_overlap(first: PurePosixPath, second: PurePosixPath) -> bool:
    return first == second or first in second.parents or second in first.parents


def _carry_bootstrap_lines(
    record: AttemptCarryForwardRecord,
    submission_id: str,
    action_root: str,
) -> tuple[str, ...]:
    sentinel = {
        "schema_version": 1,
        "phase_run_id": record.phase_run_id,
        "attempt_id": record.target_attempt_id,
        "action_id": record.workspace.target_action_id,
        "attempt_carry_forward_id": record.attempt_carry_forward_id,
        "attempt_carry_forward_digest": record.digest,
        "phase_submission_id": submission_id,
        "workspace_mapping_digest": record.workspace.digest,
    }
    encoded = base64.b64encode(
        (json.dumps(sentinel, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    ).decode()
    bootstrap = {
        "action_root": action_root,
        "workspace_root": record.workspace.workspace_root,
        "root_paths": [item.physical_root for item in record.workspace.roots],
        "sentinel_path": record.workspace.identity_sentinel_path,
        "sentinel_b64": encoded,
    }
    encoded_bootstrap = base64.b64encode(json.dumps(bootstrap, separators=(",", ":"), sort_keys=True).encode()).decode()
    program = r"""import base64
import json
import os
import stat
import sys


def reject(message):
    raise RuntimeError(message)


def normalized(value, label):
    if not isinstance(value, str) or not value.startswith("/") or os.path.normpath(value) != value:
        reject(f"{label} is not an absolute normalized path")
    return value


def open_directory_components(path):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.split("/")[1:]:
            successor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = successor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


try:
    specification = json.loads(base64.b64decode(sys.argv[1], validate=True))
    action_root = normalized(specification["action_root"], "action root")
    workspace_root = normalized(specification["workspace_root"], "workspace root")
    roots = tuple(normalized(value, "workspace bind source") for value in specification["root_paths"])
    sentinel_path = normalized(specification["sentinel_path"], "identity sentinel")
    if os.path.dirname(workspace_root) != action_root or os.path.basename(workspace_root) != "work":
        reject("carry workspace is not the exact action-root/work child")
    if len(set(roots)) != len(roots) or any(os.path.dirname(value) != workspace_root for value in roots):
        reject("workspace bind sources must be unique direct workspace children")
    if os.path.dirname(sentinel_path) != workspace_root or sentinel_path in roots:
        reject("identity sentinel is not a distinct direct workspace child")
    if os.path.realpath(action_root) != action_root:
        reject("action root escapes through a symlinked component")
    action_descriptor = open_directory_components(action_root)
    try:
        if os.path.lexists(workspace_root):
            reject("carry workspace already exists")
        os.mkdir("work", mode=0o700, dir_fd=action_descriptor)
        workspace_descriptor = os.open(
            "work", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=action_descriptor
        )
        try:
            if os.path.realpath(workspace_root) != workspace_root:
                reject("created workspace escapes its declared path")
            for path in roots:
                os.mkdir(os.path.basename(path), mode=0o700, dir_fd=workspace_descriptor)
            for path in roots:
                descriptor = os.open(
                    os.path.basename(path),
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=workspace_descriptor,
                )
                try:
                    if not stat.S_ISDIR(os.fstat(descriptor).st_mode) or os.path.realpath(path) != path:
                        reject("created bind source does not retain exact directory identity")
                finally:
                    os.close(descriptor)
            data = base64.b64decode(specification["sentinel_b64"], validate=True)
            sentinel_descriptor = os.open(
                os.path.basename(sentinel_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=workspace_descriptor,
            )
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(sentinel_descriptor, view):]
                os.fsync(sentinel_descriptor)
            finally:
                os.close(sentinel_descriptor)
            os.fsync(workspace_descriptor)
        finally:
            os.close(workspace_descriptor)
        os.fsync(action_descriptor)
    finally:
        os.close(action_descriptor)
except BaseException as error:
    print(f"carry bootstrap rejected: {error}", file=sys.stderr)
    raise SystemExit(127) from error
"""
    return (f"python3 -c {shlex.quote(program)} {shlex.quote(encoded_bootstrap)}",)


_ACCEPTANCE_CACHE_BOOTSTRAP_TARGET = "/run/bspp-acceptance-cache-root"


def _acceptance_cache_bootstrap_lines(
    staging: DatabaseProfileStagingSnapshot,
    *,
    cluster_image_path: str,
    required_mounts: tuple[PhaseContainerMountDescriptor, ...],
    required_protected_owners: tuple[tuple[str, PhaseContainerMountDescriptor], ...],
    identity_protected: tuple[str, ...],
    tolerate_failure: bool = False,
) -> tuple[str, ...]:
    """Render the in-job acceptance cache bootstrap for a staged branch.

    ``stage-required`` placement mounts the effective-user cache namespace
    (``<cache_root>/users/<unix_user>``) read-write, but nothing inside the
    action creates that directory, so the action depended on a separate
    cache-clearing job having prepared it and on it surviving between jobs.
    These lines make the action self-contained: the host-side anchor creates
    the cache root so the bootstrap mount source exists, then a container step
    creates the effective-user namespace (mode 0700) as the container's
    effective user before Database Placement runs.

    ``tolerate_failure`` renders the bootstrap best-effort (``|| true``) for
    ``stage-preferred`` profiles: a failed bootstrap must not hard-stop the
    action under ``set -e`` before placement can select the authorized
    direct-capacity fallback. ``stage-required`` keeps the bootstrap fatal
    because there is no fallback. The creation-only, idempotent,
    effective-user-owned acceptance-cache invariants are unchanged.
    """
    bootstrap_mounts = _container_mounts(
        declared=(),
        extra=(),
        required=tuple((mount.source, mount.target) for mount in required_mounts),
        database=(
            PhaseContainerMountDescriptor(
                source=staging.cache_root,
                target=_ACCEPTANCE_CACHE_BOOTSTRAP_TARGET,
                source_kind="directory",
                read_only=False,
                origin="database-protected",
            ),
        ),
        protected=("/usr/local/bin",),
        identity_protected=identity_protected,
        protected_owners=required_protected_owners,
    )
    bootstrap = _shell_command(
        (
            "srun",
            f"--container-image={cluster_image_path}",
            f"--container-mounts={bootstrap_mounts}",
            "--no-container-mount-home",
            "/usr/local/bin/entrypoint.sh",
            "mkdir",
            "-p",
            "--mode=0700",
            "--",
            f"{_ACCEPTANCE_CACHE_BOOTSTRAP_TARGET}/users/{staging.unix_user}",
        )
    )
    return (
        f"mkdir -p --mode=0700 -- {shlex.quote(staging.cache_root)}" + (" || true" if tolerate_failure else ""),
        bootstrap + (" || true" if tolerate_failure else ""),
    )


def _shell_command(argv: tuple[str, ...]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in argv)


__all__ = ["render_phase_submission_intent"]
