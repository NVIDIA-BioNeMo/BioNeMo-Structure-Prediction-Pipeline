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

"""Cluster-side seam-transport publish submission for preprocessing and folding.

The publish commands (``publish-preprocessing`` and ``publish-folding``) upload
verified MSA-set bundles or prediction bundles from cluster Lustre to S3 via
the runtime CLI's ``s5cmd`` transport.  Because the bundle lives on cluster
Lustre, the runtime CLI must execute **inside a Slurm job + the imported
runtime container** on the cluster — never the local workstation interpreter
(see ADR-0075 § "Boundary classification for the publish family").

This module renders one bounded Slurm job that runs the runtime CLI inside the
runtime container, monitors it to a terminal Slurm state, and fetches only the
small evidence JSON files back to the workstation.  The full bundle bytes stay
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from bspp.orchestration.contract.phase import (
    FoldingPhaseRunSpec,
    FoldingResolvedClusterSnapshot,
    PhaseRunSpec,
    ResolvedClusterSnapshot,
)
from bspp.orchestration.contract.runspec import SlurmResources
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore
from bspp.orchestration.control.profiles import (
    PostprocessingCredentialMountProfile,
    ResolvedClusterProfile,
)
from bspp.orchestration.control.transport import (
    TERMINAL_SLURM_STATES,
    CommandRunner,
    RemoteSlurmTransport,
    SlurmJobState,
    default_command_runner,
)

_PUBLISH_SCHEDULING_CLASS = "control_cpu"
_JOB_NAME_PREFIX = "bspp_publish_"
_ORCHESTRATION_CONTAINER_TARGET = "/workspace/bspp-orchestration"
_AWS_CONTAINER_ROOT = "/workspace/bspp-aws"
_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024  # 4 MiB bounded evidence read per file

_PREPROCESSING_EVIDENCE_FILES = (
    "artifact-location-remote.json",
    "msa-set-upload-evidence.json",
)
_FOLDING_EVIDENCE_FILES = ("prediction-bundle-upload-evidence.json",)


@dataclass(frozen=True)
class PublishPreprocessingResult:
    """Bounded publish evidence returned to the workstation."""

    job_id: str
    evidence_files: tuple[str, ...]

    def render_json(self) -> str:
        return json.dumps(
            {"job_id": self.job_id, "evidence_files": list(self.evidence_files)},
            sort_keys=True,
        )


@dataclass(frozen=True)
class PublishFoldingResult:
    """Bounded publish evidence returned to the workstation."""

    job_id: str
    evidence_files: tuple[str, ...]

    def render_json(self) -> str:
        return json.dumps(
            {"job_id": self.job_id, "evidence_files": list(self.evidence_files)},
            sort_keys=True,
        )


def publish_preprocessing_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    handoff_path: Path,
    evidence_dir: Path,
    cluster_profile: ResolvedClusterProfile,
    runner: CommandRunner = default_command_runner,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 3600.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> PublishPreprocessingResult:
    """Submit, monitor, and fetch evidence for one preprocessing publish job.

    The runtime CLI reads the staged input JSON (containing the artifact
    location and s3 prefix), reads the bundle from Lustre, uploads to S3 via
    s5cmd, and writes evidence JSON to the cluster evidence directory.  This
    function stages the input, renders and submits the sbatch script, monitors
    to terminal, and fetches the evidence files back.
    """
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    runspec = authority.phase_runspec
    if not isinstance(runspec, PhaseRunSpec):
        raise ValueError("authority is not a preprocessing phase run")
    if runspec.payload.transport != "publish-to-s3":
        raise ValueError(f"preprocessing RunSpec transport is {runspec.payload.transport!r}, not 'publish-to-s3'")
    if runspec.payload.s3_publish_prefix is None:
        raise ValueError("preprocessing RunSpec has no s3_publish_prefix")

    _validate_profile_matches_runspec(cluster_profile, runspec)

    artifact_location_path = handoff_path / "artifact-location.json"
    if not artifact_location_path.is_file():
        raise ValueError(f"missing artifact-location.json in handoff: {artifact_location_path}")
    artifact_location_mapping = json.loads(artifact_location_path.read_text())

    input_payload = {
        "artifact_location": artifact_location_mapping,
        "s3_prefix": runspec.payload.s3_publish_prefix,
        "phase_run_id": phase_run_id,
        "attempt_id": runspec.attempt_id,
    }

    _reject_evidence_inside_authority(evidence_dir, authority_root)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = _extract_bundle_path(artifact_location_mapping)
    result = _submit_publish_job(
        phase_kind="preprocessing",
        input_payload=input_payload,
        bundle_paths=(bundle_path,) if bundle_path else (),
        evidence_dir=evidence_dir,
        cluster_profile=cluster_profile,
        runtime_image=runspec.cluster.runtime_image,
        evidence_filenames=_PREPROCESSING_EVIDENCE_FILES,
        runner=runner,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        sleeper=sleeper,
    )
    return PublishPreprocessingResult(
        job_id=result.job_id,
        evidence_files=result.evidence_files,
    )


def publish_folding_phase(
    phase_run_id: str,
    *,
    authority_root: Path,
    bundles_path: Path,
    local_paths_json: Path,
    evidence_dir: Path,
    cluster_profile: ResolvedClusterProfile,
    runner: CommandRunner = default_command_runner,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 3600.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> PublishFoldingResult:
    """Submit, monitor, and fetch evidence for one folding publish job.

    The runtime CLI reads the staged input JSON (containing operator-attested
    prediction bundle records, local paths, and s3 prefix), reads each bundle
    from Lustre, uploads to S3 via s5cmd, and writes evidence JSON to the
    cluster evidence directory.
    """
    store = PhaseAuthorityStore(authority_root)
    authority = store.validate(phase_run_id)
    runspec = authority.phase_runspec
    if not isinstance(runspec, FoldingPhaseRunSpec):
        raise ValueError("authority is not a folding phase run")
    if runspec.payload.transport != "publish-to-s3":
        raise ValueError(f"folding RunSpec transport is {runspec.payload.transport!r}, not 'publish-to-s3'")
    if runspec.payload.s3_prediction_prefix is None:
        raise ValueError("folding RunSpec has no s3_prediction_prefix")

    _validate_profile_matches_runspec(cluster_profile, runspec)

    bundles_data = json.loads(bundles_path.read_text())
    local_paths = json.loads(local_paths_json.read_text())

    input_payload = {
        "bundles": bundles_data,
        "local_paths": local_paths,
        "s3_prefix": runspec.payload.s3_prediction_prefix,
        "phase_run_id": phase_run_id,
        "attempt_id": runspec.attempt_id,
    }

    _reject_evidence_inside_authority(evidence_dir, authority_root)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    bundle_paths = tuple(Path(p) for p in local_paths) if local_paths else ()
    result = _submit_publish_job(
        phase_kind="folding",
        input_payload=input_payload,
        bundle_paths=bundle_paths,
        evidence_dir=evidence_dir,
        cluster_profile=cluster_profile,
        runtime_image=runspec.cluster.runtime_image,
        evidence_filenames=_FOLDING_EVIDENCE_FILES,
        runner=runner,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        sleeper=sleeper,
    )
    return PublishFoldingResult(
        job_id=result.job_id,
        evidence_files=result.evidence_files,
    )


@dataclass(frozen=True)
class _PublishJobResult:
    job_id: str
    evidence_files: tuple[str, ...]


def _submit_publish_job(
    *,
    phase_kind: str,
    input_payload: dict[str, object],
    bundle_paths: tuple[Path, ...],
    evidence_dir: Path,
    cluster_profile: ResolvedClusterProfile,
    runtime_image: str,
    evidence_filenames: tuple[str, ...],
    runner: CommandRunner,
    poll_interval_seconds: float,
    timeout_seconds: float,
    sleeper: Callable[[float], None],
) -> _PublishJobResult:
    resource = cluster_profile.resources.get(_PUBLISH_SCHEDULING_CLASS)
    if resource is None:
        raise ValueError(f"Cluster Profile {cluster_profile.name!r} requires resources.{_PUBLISH_SCHEDULING_CLASS}")

    credential_mounts = cluster_profile.postprocessing_credential_mounts
    if credential_mounts is None:
        raise ValueError(
            "publish requires Cluster Profile postprocessing_credential_mounts "
            "with both cluster-visible shared-profile files"
        )

    transport = RemoteSlurmTransport(
        kind=cluster_profile.transport,
        ssh_target=cluster_profile.ssh_target,
        runner=runner,
    )

    staging_root = Path(cluster_profile.staging_root)
    job_token = hashlib.sha256(json.dumps(input_payload, sort_keys=True).encode()).hexdigest()[:12]
    job_name = _JOB_NAME_PREFIX + job_token

    cluster_input_path = staging_root / "bspp-publish" / f"{job_token}" / "input.json"
    cluster_evidence_dir = staging_root / "bspp-publish" / f"{job_token}" / "evidence"
    cluster_log_dir = cluster_evidence_dir / "slurm-logs"

    if cluster_profile.transport == "ssh":
        for d in (cluster_input_path.parent, cluster_evidence_dir, cluster_log_dir):
            result = transport.command(("mkdir", "-p", str(d)))
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "mkdir failed"
                raise ValueError(f"failed to create publish directory {d}: {detail}")
    else:
        for d in (cluster_input_path.parent, cluster_evidence_dir, cluster_log_dir):
            d.mkdir(parents=True, exist_ok=True)

    input_json_bytes = (json.dumps(input_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    input_sha256 = hashlib.sha256(input_json_bytes).hexdigest()

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False, dir=str(evidence_dir)) as f:
        f.write(input_json_bytes)
        local_input_path = Path(f.name)

    try:
        if cluster_profile.transport == "ssh":
            transport.stage_immutable_artifact(
                local_input_path,
                str(cluster_input_path),
                expected_sha256=input_sha256,
                staging_token=f"publish-input-{job_token}",
            )
        else:
            cluster_input_path.write_bytes(input_json_bytes)
    finally:
        local_input_path.unlink(missing_ok=True)

    script = _render_publish_script(
        phase_kind=phase_kind,
        job_name=job_name,
        resource=resource,
        cluster_profile=cluster_profile,
        runtime_image=runtime_image,
        cluster_input_path=cluster_input_path,
        cluster_evidence_dir=cluster_evidence_dir,
        cluster_log_dir=cluster_log_dir,
        bundle_paths=bundle_paths,
        credential_mounts=credential_mounts,
    )

    local_script_path = evidence_dir / f"{job_name}.sbatch"
    local_script_path.write_text(script, encoding="utf-8")
    script_sha256 = hashlib.sha256(script.encode()).hexdigest()

    if cluster_profile.transport == "ssh":
        cluster_script_path = cluster_evidence_dir / f"{job_name}.sbatch"
        transport.stage_immutable_artifact(
            local_script_path,
            str(cluster_script_path),
            expected_sha256=script_sha256,
            staging_token=f"publish-script-{job_token}",
        )
        submission = transport.submit_script(cluster_script_path, job_name=job_name)
    else:
        submission = transport.submit_script(local_script_path, job_name=job_name)

    state = _wait_for_terminal_state(
        transport,
        submission.job_id,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        sleeper=sleeper,
    )
    if state.state != "COMPLETED" or state.exit_code != "0:0":
        raise ValueError(
            f"publish job {submission.job_id} ended in state {state.state!r} "
            f"with exit code {state.exit_code!r}; expected COMPLETED with exit code 0:0"
        )

    fetched = _fetch_evidence(
        transport,
        cluster_evidence_dir,
        evidence_dir,
        evidence_filenames,
    )
    return _PublishJobResult(job_id=submission.job_id, evidence_files=fetched)


def _render_publish_script(
    *,
    phase_kind: str,
    job_name: str,
    resource: SlurmResources,
    cluster_profile: ResolvedClusterProfile,
    runtime_image: str,
    cluster_input_path: Path,
    cluster_evidence_dir: Path,
    cluster_log_dir: Path,
    bundle_paths: tuple[Path, ...],
    credential_mounts: PostprocessingCredentialMountProfile,
) -> str:
    # Mount polarity mirrors the sibling convention (folding_benchmark_submit):
    # inputs, credentials, source repos, and bundle parents are read-only (False);
    # the evidence directory — the only writable output — is read-write (True).
    mounts: list[tuple[str, str, bool]] = [
        (str(cluster_input_path), str(cluster_input_path), False),
        (str(cluster_evidence_dir), str(cluster_evidence_dir), True),
        (str(cluster_profile.orchestration_repo), _ORCHESTRATION_CONTAINER_TARGET, False),
        (credential_mounts.aws_shared_credentials_file, f"{_AWS_CONTAINER_ROOT}/credentials", False),
        (credential_mounts.aws_config_file, f"{_AWS_CONTAINER_ROOT}/config", False),
    ]
    for bundle in bundle_paths:
        if bundle.is_absolute():
            parent = str(PurePosixPath(bundle).parent)
            mounts.append((parent, parent, False))

    mounts = list(_dedupe_mounts(mounts))
    mounts_arg = ",".join(_render_mount(mount) for mount in mounts)

    runtime_command = (
        "bspp-orchestration-runtime",
        "phase",
        f"publish-{phase_kind}",
        "--input-json",
        str(cluster_input_path),
        "--evidence-dir",
        str(cluster_evidence_dir),
    )
    action_command = " ".join(shlex.quote(part) for part in runtime_command)

    srun = [
        "srun",
        f"--container-image={runtime_image}",
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
        action_command,
    ]
    token = job_name.removeprefix(_JOB_NAME_PREFIX)
    lines = [
        "#!/usr/bin/env bash",
        f"# BSPP seam-transport publish ({phase_kind})",
        f"# Job name: {job_name}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resource.partition}",
        f"#SBATCH --account={cluster_profile.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resource.cpus_per_task}",
        f"#SBATCH --mem={resource.memory}",
        f"#SBATCH --time={resource.time}",
        f"#SBATCH --output={cluster_log_dir / f'publish.{token}.%j.out'}",
        f"#SBATCH --error={cluster_log_dir / f'publish.{token}.%j.err'}",
        "",
        "set -euo pipefail",
        "",
        srun_line + " <<'BSPP_PUBLISH'",
        *heredoc,
        "BSPP_PUBLISH",
        "",
    ]
    return "\n".join(lines)


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
            raise ValueError(f"timed out waiting for publish job {job_id} to reach a terminal Slurm state")
        sleeper(poll_interval_seconds)


def _fetch_evidence(
    transport: RemoteSlurmTransport,
    cluster_evidence_dir: Path,
    local_evidence_dir: Path,
    expected_filenames: tuple[str, ...],
) -> tuple[str, ...]:
    fetched: list[str] = []
    for filename in expected_filenames:
        cluster_path = cluster_evidence_dir / filename
        local_path = local_evidence_dir / filename
        with tempfile.TemporaryDirectory(prefix="bspp-publish-fetch-") as tmp_dir:
            tmp_path = Path(tmp_dir) / filename
            result = transport.fetch_artifact(str(cluster_path), tmp_path)
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "evidence fetch failed"
                raise ValueError(f"failed to fetch publish evidence {cluster_path}: {detail}")
            if not tmp_path.is_file() or tmp_path.is_symlink():
                raise ValueError(f"publish evidence fetch did not produce a regular file: {filename}")
            if tmp_path.stat().st_size > _MAX_EVIDENCE_BYTES:
                raise ValueError(f"publish evidence file exceeds bounded read limit: {filename}")
            data = tmp_path.read_bytes()
            try:
                json.loads(data)
            except ValueError as exc:
                raise ValueError(f"publish evidence file is not valid JSON: {filename}: {exc}") from exc
            local_path.write_bytes(data)
            fetched.append(filename)
    return tuple(fetched)


def _validate_profile_matches_runspec(
    cluster_profile: ResolvedClusterProfile,
    runspec: PhaseRunSpec | FoldingPhaseRunSpec,
) -> None:
    """Refuse a profile that does not match the attempt's frozen cluster snapshot.

    Publish must never submit to a different cluster, account, transport, or
    staging root than the one frozen at materialization.  The profile name is
    the primary check; transport, ssh_target, account, staging_root, and
    orchestration_repo are belt-and-suspenders fields that must match because
    they come from the same resolved profile.
    """
    snapshot = runspec.cluster
    mismatches: list[str] = []
    if cluster_profile.name != snapshot.profile_name:
        mismatches.append(f"profile name {cluster_profile.name!r} != frozen {snapshot.profile_name!r}")
    if cluster_profile.transport != snapshot.transport:
        mismatches.append(f"transport {cluster_profile.transport!r} != frozen {snapshot.transport!r}")
    if cluster_profile.ssh_target != snapshot.ssh_target:
        mismatches.append(f"ssh_target {cluster_profile.ssh_target!r} != frozen {snapshot.ssh_target!r}")
    if cluster_profile.account != snapshot.account:
        mismatches.append(f"account {cluster_profile.account!r} != frozen {snapshot.account!r}")
    if cluster_profile.staging_root != snapshot.staging_root:
        mismatches.append(f"staging_root {cluster_profile.staging_root!r} != frozen {snapshot.staging_root!r}")
    if cluster_profile.orchestration_repo != snapshot.orchestration_repo:
        mismatches.append(
            f"orchestration_repo {cluster_profile.orchestration_repo!r} != frozen {snapshot.orchestration_repo!r}"
        )
    # Image identity check — branch on snapshot type because the frozen
    # runtime_image is sourced differently for each phase kind.
    #   Preprocessing: snapshot.runtime_image = qualified_runtime's
    #     cluster_image_path (phase_attempt_materialization.py:169-180),
    #     NOT profile.image.  Compare the same field pair the materialization
    #     froze: profile.preprocessing_runtime.cluster_image_path vs snapshot.
    #   Folding: snapshot.runtime_image = profile.image
    #     (phase_materialization.py:381).  Compare profile.image directly.
    if isinstance(snapshot, ResolvedClusterSnapshot):
        prep_runtime = cluster_profile.preprocessing_runtime
        if prep_runtime is None:
            mismatches.append("profile has no preprocessing_runtime but runspec is a preprocessing phase run")
        elif prep_runtime.cluster_image_path != snapshot.runtime_image:
            mismatches.append(
                f"preprocessing_runtime.cluster_image_path {prep_runtime.cluster_image_path!r}"
                f" != frozen runtime_image {snapshot.runtime_image!r}"
            )
    elif isinstance(snapshot, FoldingResolvedClusterSnapshot):
        if cluster_profile.image != snapshot.runtime_image:
            mismatches.append(f"image {cluster_profile.image!r} != frozen runtime_image {snapshot.runtime_image!r}")
    if mismatches:
        raise ValueError(
            f"Cluster Profile {cluster_profile.name!r} does not match the frozen RunSpec cluster snapshot: "
            + "; ".join(mismatches)
        )


def _extract_bundle_path(artifact_location_mapping: dict[str, object]) -> Path | None:
    raw = artifact_location_mapping.get("bundle_path")
    if isinstance(raw, str) and raw:
        return Path(raw)
    return None


def _reject_evidence_inside_authority(evidence_dir: Path, authority_root: Path) -> None:
    evidence_resolved = evidence_dir.resolve()
    authority_resolved = authority_root.resolve()
    if evidence_resolved == authority_resolved or str(evidence_resolved).startswith(str(authority_resolved) + "/"):
        raise ValueError("--evidence-dir must not resolve inside the authority root")


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


__all__ = [
    "PublishFoldingResult",
    "PublishPreprocessingResult",
    "publish_folding_phase",
    "publish_preprocessing_phase",
]
