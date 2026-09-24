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

"""Workstation-side submission for the legacy-MSA import job.

This module renders and submits one bounded Slurm job that runs the runtime
``folding legacy-msa-import`` worker inside the runtime container, monitors it
to a terminal Slurm state, and fetches only the worker's three small enriched
handoff records back to the workstation.  The legacy records and the payload
bundle/tar are never fetched to the workstation and never modified.

It deliberately imports only the standard library, the shared contract, and
``bspp.orchestration.control.*`` so the Control Plane import-boundary gate
remains green (no runtime, numpy, or pyarrow imports).
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    PreprocessingContentValidationEvidence,
    PreprocessingHandoffBundle,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_manifest_from_mapping,
    msa_chunk_manifest_from_mapping,
    preprocessing_content_validation_evidence_from_mapping,
    verified_local_bundled_artifact_location_from_mapping,
)
from bspp.orchestration.control.profiles import ResolvedClusterProfile
from bspp.orchestration.control.transport import (
    TERMINAL_SLURM_STATES,
    CommandRunner,
    RemoteSlurmTransport,
    SlurmJobState,
    default_command_runner,
)

_SCHEDULING_CLASS = "control_cpu"
_MAX_RECORD_BYTES = 16 * 1024 * 1024  # 16 MiB per metadata record; payloads stay in Runtime.
_JOB_NAME_PREFIX = "bspp_legacy_msa_import_"
_ORCHESTRATION_CONTAINER_TARGET = "/workspace/bspp-orchestration"


@dataclass(frozen=True)
class RenderedLegacyMsaImportSubmission:
    """One rendered legacy-MSA import Slurm submission."""

    script_path: Path
    cluster_script_path: Path
    result_dir: Path
    job_name: str
    action_command: str


@dataclass(frozen=True)
class LegacyMsaImportResult:
    """Bounded enriched-handoff identity returned to the workstation."""

    job_id: str
    artifact_set_id: str
    artifact_location_id: str
    member_lengths: tuple[int, ...]
    artifact_set: MsaArtifactSetManifest
    artifact_location: VerifiedLocalBundledArtifactLocation
    content_validation: PreprocessingContentValidationEvidence

    def render_json(self) -> str:
        """Render the bounded enriched identity as stable JSON."""
        return json.dumps(
            {
                "job_id": self.job_id,
                "artifact_set_id": self.artifact_set_id,
                "artifact_location_id": self.artifact_location_id,
                "member_lengths": list(self.member_lengths),
            },
            sort_keys=True,
        )


@dataclass(frozen=True)
class _PreparedImport:
    """Legacy handoff facts discovered by the bounded prepare step."""

    bundle: PreprocessingHandoffBundle
    tar_path: Path
    bundle_path: Path


def render_legacy_msa_import_submission(
    *,
    cluster_profile: ResolvedClusterProfile,
    handoff_root: Path,
    output_dir: Path,
    script_path: Path,
    payload_paths: tuple[Path, Path],
    lz4_executable: str = "lz4",
) -> RenderedLegacyMsaImportSubmission:
    """Render a bounded Slurm submission for the runtime legacy-MSA import worker."""
    if not lz4_executable:
        raise ValueError("legacy MSA import lz4 executable must be non-empty")
    for name, path in (("handoff-root", handoff_root), ("output-dir", output_dir)):
        if not path.is_absolute():
            raise ValueError(f"legacy MSA import {name} must be an absolute cluster path")
    tar_path, bundle_path = payload_paths
    for name, path in (("tar-path", tar_path), ("bundle-path", bundle_path)):
        if not path.is_absolute():
            raise ValueError(f"legacy MSA import {name} must be an absolute cluster path")
    resource = cluster_profile.resources.get(_SCHEDULING_CLASS)
    if resource is None:
        raise ValueError(f"Cluster Profile {cluster_profile.name!r} requires resources.{_SCHEDULING_CLASS}")

    job_name = _job_name(str(handoff_root))
    run_dir = _slurm_run_dir(output_dir)
    mounts = _dedupe_mounts(
        [
            (str(handoff_root), str(handoff_root), False),
            (str(output_dir), str(output_dir), True),
            (str(cluster_profile.orchestration_repo), _ORCHESTRATION_CONTAINER_TARGET, False),
            (str(tar_path.parent), str(tar_path.parent), False),
            (str(bundle_path.parent), str(bundle_path.parent), False),
        ]
    )
    mounts_arg = ",".join(_render_mount(mount) for mount in mounts)
    action_command = " ".join(
        (
            "bspp-orchestration-runtime",
            "folding",
            "legacy-msa-import",
            "--handoff-root",
            shlex.quote(str(handoff_root)),
            "--output-dir",
            shlex.quote(str(output_dir)),
            "--lz4",
            shlex.quote(lz4_executable),
        )
    )
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
    lines = [
        "#!/usr/bin/env bash",
        "# BSPP legacy MSA import",
        f"# Job name: {job_name}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resource.partition}",
        f"#SBATCH --account={cluster_profile.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resource.cpus_per_task}",
        f"#SBATCH --mem={resource.memory}",
        f"#SBATCH --time={resource.time}",
        f"#SBATCH --output={run_dir / 'legacy_msa_import.%j.out'}",
        f"#SBATCH --error={run_dir / 'legacy_msa_import.%j.err'}",
        "",
        "set -euo pipefail",
        "",
        srun_line + " <<'BSPP_LEGACY_MSA_IMPORT'",
        action_command,
        "BSPP_LEGACY_MSA_IMPORT",
        "",
    ]
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines), encoding="utf-8")
    cluster_script_path = run_dir / "legacy-msa-import.sbatch" if cluster_profile.transport == "ssh" else script_path
    return RenderedLegacyMsaImportSubmission(
        script_path=script_path,
        cluster_script_path=cluster_script_path,
        result_dir=output_dir,
        job_name=job_name,
        action_command=action_command,
    )


def submit_legacy_msa_import(
    *,
    cluster_profile: ResolvedClusterProfile,
    handoff_root: Path,
    output_dir: Path,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 86400.0,
    runner: CommandRunner = default_command_runner,
    sleeper: Callable[[float], None] = time.sleep,
    lz4_executable: str = "lz4",
) -> LegacyMsaImportResult:
    """Prepare, render, submit, monitor, and fetch bounded evidence for one import."""
    transport = RemoteSlurmTransport(
        kind=cluster_profile.transport,
        ssh_target=cluster_profile.ssh_target,
        runner=runner,
    )
    prepared = _prepare(transport, handoff_root)
    job_name = _job_name(str(handoff_root))
    run_dir = _slurm_run_dir(output_dir)
    local_script_path = _local_script_path(cluster_profile, run_dir, job_name=job_name)
    local_script_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_legacy_msa_import_submission(
        cluster_profile=cluster_profile,
        handoff_root=handoff_root,
        output_dir=output_dir,
        script_path=local_script_path,
        payload_paths=(prepared.tar_path, prepared.bundle_path),
        lz4_executable=lz4_executable,
    )
    # The runtime worker accepts an existing-but-empty ``output_dir`` (the
    # writable container mount source must pre-exist), while the staged sbatch
    # and Slurm logs live in the sibling ``run_dir`` so they never collide with
    # the worker's publication target.
    if cluster_profile.transport == "ssh":
        for directory in (output_dir, run_dir):
            mkdir_result = transport.command(("mkdir", "-p", str(directory)))
            if mkdir_result.returncode != 0:
                detail = mkdir_result.stderr.strip() or mkdir_result.stdout.strip() or "mkdir failed"
                raise ValueError(f"failed to create legacy MSA import directory {directory}: {detail}")
        transport.stage_immutable_artifact(
            rendered.script_path,
            str(rendered.cluster_script_path),
            expected_sha256=_file_sha256(rendered.script_path),
            staging_token=f"legacy-msa-import-{job_name.removeprefix(_JOB_NAME_PREFIX)}",
        )
        submission = transport.submit_script(rendered.cluster_script_path)
    else:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"failed to create legacy MSA import directories: {exc}") from exc
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
            f"legacy MSA import job {submission.job_id} ended in state {state.state!r} "
            f"with exit code {state.exit_code!r}; expected COMPLETED with exit code 0:0"
        )
    return _fetch_and_validate_result(transport, output_dir, prepared.bundle, submission.job_id)


def resume_legacy_msa_import(
    *,
    job_id: str,
    cluster_profile: ResolvedClusterProfile,
    handoff_root: Path,
    output_dir: Path,
    poll_interval_seconds: float = 30.0,
    timeout_seconds: float = 86400.0,
    runner: CommandRunner = default_command_runner,
    sleeper: Callable[[float], None] = time.sleep,
    lz4_executable: str = "lz4",
) -> LegacyMsaImportResult:
    """Verify and monitor one existing scalar import; never stage or submit.

    Identity comes from current accounting's exact SubmitLine and the original
    immutable staged script, not from job-name discovery or a saved CLI result.
    The staged file is not claimed to be Slurm's retained historical script.
    """
    if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
        raise ValueError("legacy MSA import resume requires one canonical positive scalar job id")
    transport = RemoteSlurmTransport(
        kind=cluster_profile.transport, ssh_target=cluster_profile.ssh_target, runner=runner
    )
    prepared = _prepare(transport, handoff_root)
    # Render only in a private workstation directory. The original cluster
    # script has this fixed sibling path for both supported transports.
    script_path = _slurm_run_dir(output_dir) / "legacy-msa-import.sbatch"
    with tempfile.TemporaryDirectory(prefix="bspp-legacy-msa-resume-") as tmp_dir:
        rendered = render_legacy_msa_import_submission(
            cluster_profile=cluster_profile,
            handoff_root=handoff_root,
            output_dir=output_dir,
            script_path=Path(tmp_dir) / "expected.sbatch",
            payload_paths=(prepared.tar_path, prepared.bundle_path),
            lz4_executable=lz4_executable,
        )
        expected_script = rendered.script_path.read_bytes()
    _verify_existing_import(transport, job_id, cluster_profile.owner, rendered.job_name, script_path, expected_script)
    state = _wait_for_terminal_state(
        transport,
        job_id,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        sleeper=sleeper,
        best_effort=True,
    )
    if state.job_id != job_id or state.source != "sacct" or state.state != "COMPLETED" or state.exit_code != "0:0":
        raise ValueError(
            f"legacy MSA import job {job_id} requires exact sacct COMPLETED with exit code 0:0; "
            f"observed {state.to_mapping()}"
        )
    _verify_existing_import(transport, job_id, cluster_profile.owner, rendered.job_name, script_path, expected_script)
    return _fetch_and_validate_result(transport, output_dir, prepared.bundle, job_id)


def _verify_existing_import(
    transport: RemoteSlurmTransport,
    job_id: str,
    owner: str,
    job_name: str,
    script_path: Path,
    expected_script: bytes,
) -> None:
    owner_result = transport.command(("id", "-un"))
    if owner_result.returncode != 0 or owner_result.stdout.strip() != owner:
        raise ValueError("legacy MSA import resume owner does not match the Cluster Profile")
    identity = transport.command(
        (
            "env",
            "TZ=UTC",
            "sacct",
            "--duplicates",
            "-X",
            "--parsable2",
            "--noheader",
            "--jobs",
            job_id,
            "--format=JobIDRaw,JobName%128,User%128,SubmitLine%4096",
        )
    )
    if identity.returncode != 0 or len(identity.stdout.encode("utf-8")) > 65536:
        raise ValueError("legacy MSA import resume scheduler identity is unavailable or exceeds its bound")
    rows = [line.split("|") for line in identity.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][:3] != [job_id, job_name, owner]:
        raise ValueError("legacy MSA import resume requires one exact job/name/owner accounting row")
    try:
        submit_argv = shlex.split(rows[0][3])
    except ValueError as exc:
        raise ValueError("legacy MSA import resume has a malformed SubmitLine") from exc
    if submit_argv != ["sbatch", "--parsable", str(script_path)]:
        raise ValueError("legacy MSA import resume SubmitLine does not match the original script invocation")
    actual_script = transport.fetch_stable_artifact(
        script_path, remote_root=script_path.parent, maximum_bytes=256 * 1024
    )
    if actual_script != expected_script:
        raise ValueError("legacy MSA import resume staged script differs from the public renderer")


def _prepare(transport: RemoteSlurmTransport, handoff_root: Path) -> _PreparedImport:
    if not handoff_root.is_absolute():
        raise ValueError("legacy MSA import handoff-root must be an absolute cluster path")
    artifact_set = msa_artifact_set_manifest_from_mapping(_fetch_record(transport, handoff_root / "artifact-set.json"))
    chunk_relative = Path(artifact_set.chunks[0].logical_path)
    bundle = PreprocessingHandoffBundle(
        chunk_manifest=msa_chunk_manifest_from_mapping(_fetch_record(transport, handoff_root / chunk_relative)),
        artifact_set=artifact_set,
        artifact_location=verified_local_bundled_artifact_location_from_mapping(
            _fetch_record(transport, handoff_root / "artifact-location.json")
        ),
        content_validation=preprocessing_content_validation_evidence_from_mapping(
            _fetch_record(transport, handoff_root / "content-validation.json")
        ),
    )
    if bundle.artifact_set.has_member_lengths():
        raise ValueError("legacy MSA handoff is already enriched")
    return _PreparedImport(
        bundle=bundle,
        tar_path=Path(bundle.artifact_location.tar_path),
        bundle_path=Path(bundle.artifact_location.bundle_path),
    )


def _fetch_record(transport: RemoteSlurmTransport, remote_path: Path) -> Mapping[str, object]:
    with tempfile.TemporaryDirectory(prefix="bspp-legacy-msa-import-") as tmp_dir:
        local_path = Path(tmp_dir) / "record.json"
        result = transport.fetch_artifact(str(remote_path), local_path)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "record fetch failed"
            raise ValueError(f"failed to fetch legacy MSA handoff record {remote_path}: {detail}")
        if not local_path.is_file() or local_path.is_symlink():
            raise ValueError("legacy MSA handoff record fetch did not produce a regular file")
        if local_path.stat().st_size > _MAX_RECORD_BYTES:
            raise ValueError("legacy MSA handoff record exceeds the bounded read limit")
        try:
            payload = json.loads(local_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"legacy MSA handoff record is missing or invalid: {remote_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("legacy MSA handoff record must be a JSON object")
        return payload


def _fetch_and_validate_result(
    transport: RemoteSlurmTransport,
    output_dir: Path,
    legacy: PreprocessingHandoffBundle,
    job_id: str,
) -> LegacyMsaImportResult:
    enriched = msa_artifact_set_manifest_from_mapping(_fetch_record(transport, output_dir / "artifact-set.json"))
    rebound_location = verified_local_bundled_artifact_location_from_mapping(
        _fetch_record(transport, output_dir / "artifact-location.json")
    )
    rebound_validation = preprocessing_content_validation_evidence_from_mapping(
        _fetch_record(transport, output_dir / "content-validation.json")
    )
    if enriched.member_lengths is None:
        raise ValueError("enriched MSA Artifact Set manifest is missing member_lengths")
    if len(enriched.member_lengths) != enriched.member_count:
        raise ValueError("enriched MSA Artifact Set member_lengths count must match member_count")
    if any(length <= 0 for length in enriched.member_lengths):
        raise ValueError("enriched MSA Artifact Set member_lengths must be positive integers")
    if (
        enriched.chunks != legacy.artifact_set.chunks
        or enriched.member_count != legacy.artifact_set.member_count
        or enriched.logical_bytes != legacy.artifact_set.logical_bytes
    ):
        raise ValueError("enriched MSA Artifact Set must preserve the legacy chunk reference and logical content")
    legacy_location = legacy.artifact_location
    if (
        rebound_location.tar_path != legacy_location.tar_path
        or rebound_location.bundle_path != legacy_location.bundle_path
        or rebound_location.bundle_uri != legacy_location.bundle_uri
        or rebound_location.tar_size_bytes != legacy_location.tar_size_bytes
        or rebound_location.tar_sha256 != legacy_location.tar_sha256
        or rebound_location.lz4_size_bytes != legacy_location.lz4_size_bytes
        or rebound_location.lz4_sha256 != legacy_location.lz4_sha256
        or rebound_location.raw_tar_members != legacy_location.raw_tar_members
        or rebound_location.members != legacy_location.members
    ):
        raise ValueError("rebound MSA Artifact Location must preserve the legacy payload byte references")
    if rebound_validation.artifact_set_manifest_digest != canonical_mapping_digest(enriched.to_mapping()):
        raise ValueError("rebound content validation must bind the exact enriched Artifact Set manifest")
    # Reconstructing the bundle re-runs the cross-record binding checks.
    PreprocessingHandoffBundle(
        chunk_manifest=legacy.chunk_manifest,
        artifact_set=enriched,
        artifact_location=rebound_location,
        content_validation=rebound_validation,
    )
    return LegacyMsaImportResult(
        job_id=job_id,
        artifact_set_id=enriched.artifact_set_id,
        artifact_location_id=rebound_location.artifact_location_id,
        member_lengths=enriched.member_lengths,
        artifact_set=enriched,
        artifact_location=rebound_location,
        content_validation=rebound_validation,
    )


def _wait_for_terminal_state(
    transport: RemoteSlurmTransport,
    job_id: str,
    *,
    poll_interval_seconds: float,
    timeout_seconds: float,
    sleeper: Callable[[float], None],
    best_effort: bool = False,
) -> SlurmJobState:
    deadline = time.monotonic() + timeout_seconds
    while True:
        query = transport.query_observation_best_effort if best_effort else transport.query_observation
        observation = query((job_id,), require_exact_terminal_exit=True)
        states = observation.selected_states
        if states:
            state = states[0]
            if state.state in TERMINAL_SLURM_STATES:
                return state
        if time.monotonic() >= deadline:
            raise ValueError(f"timed out waiting for legacy MSA import job {job_id} to reach a terminal Slurm state")
        sleeper(poll_interval_seconds)


def _job_name(handoff_root: str) -> str:
    return _JOB_NAME_PREFIX + hashlib.sha256(handoff_root.encode("utf-8")).hexdigest()[:12]


def _slurm_run_dir(output_dir: Path) -> Path:
    """Sibling directory holding the staged sbatch and Slurm logs.

    The runtime worker treats ``output_dir`` as its create-or-empty publication
    target, so every submission artifact (the staged sbatch script and the
    ``--output``/``--error`` logs) must live outside it.
    """
    return output_dir.parent / f"{output_dir.name}-slurm"


def _local_script_path(
    cluster_profile: ResolvedClusterProfile,
    run_dir: Path,
    *,
    job_name: str,
) -> Path:
    if cluster_profile.transport == "ssh":
        return Path.cwd() / ".bspp-legacy-msa-import" / f"{job_name}.sbatch"
    return run_dir / "legacy-msa-import.sbatch"


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
    "LegacyMsaImportResult",
    "RenderedLegacyMsaImportSubmission",
    "render_legacy_msa_import_submission",
    "resume_legacy_msa_import",
    "submit_legacy_msa_import",
]
