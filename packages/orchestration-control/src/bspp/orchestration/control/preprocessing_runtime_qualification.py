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

"""Scheduled qualification for the dedicated preprocessing runtime image."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import yaml

from bspp.orchestration.contract.database_placement import SELECTED_DATABASE_ROOT
from bspp.orchestration.contract.preprocessing_runtime import (
    PREPROCESSING_ADAPTER_VERSION,
    PREPROCESSING_RUNTIME_CONTRACT_ID,
    PreprocessingRuntimeImageIdentity,
    PreprocessingRuntimeQualificationRecord,
    PreprocessingRuntimeQualificationTuple,
    QualifiedPreprocessingRuntimeSelection,
    preprocessing_runtime_qualification_record_from_mapping,
    preprocessing_runtime_tuple_id,
)
from bspp.orchestration.control.profiles import ResolvedClusterProfile, resolve_cluster_profile
from bspp.orchestration.control.transport import CommandRunner, RemoteSlurmTransport, default_command_runner

_SCHEDULING_CLASS = "gpu_worker"
_IMAGE_SMOKE = "/opt/bspp/bin/bspp-preprocessing-image-smoke"


class PreprocessingRuntimeQualificationError(ValueError):
    """The preprocessing runtime cannot be qualified or selected."""


def derive_smoke_gres(
    *,
    gres: str | None,
    nodes: int | None,
    gpus_per_task: int | None,
    profile_name: str,
) -> str:
    """Return the single-GPU ``--gres`` string for the qualification smoke.

    Legacy/unpacked profiles carry an explicit ``gres`` string (e.g. ``gpu:1``);
    it is returned byte-identical so existing qualifications never regress.
    Packed-topology profiles (``nodes`` + ``gpus_per_task``, no
    ``gres``) derive ``gpu:<gpus_per_task>`` — the smoke is single-GPU by design
    and packed profiles set ``gpus_per_task: 1``.  A profile with neither a legacy
    ``gres`` nor a typed topology has no GPU request and must not qualify.
    """
    if gres is not None:
        return gres
    if nodes is not None and gpus_per_task is not None:
        return f"gpu:{gpus_per_task}"
    raise PreprocessingRuntimeQualificationError(
        f"Cluster Profile {profile_name!r} requires resources.{_SCHEDULING_CLASS}.gres "
        f"or typed topology (nodes + gpus_per_task)"
    )


def qualify_preprocessing_runtime(
    *,
    profile_name: str,
    config_path: Path,
    source_repo: Path,
    now: datetime | None = None,
    runner: CommandRunner = default_command_runner,
) -> PreprocessingRuntimeQualificationRecord:
    """Persist intent, then submit the only job allowed to qualify the tuple."""
    profile = resolve_cluster_profile(profile_name, config_path=config_path)
    qualification_tuple = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(qualification_tuple)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    authority_path = _preprocessing_runtime_qualification_authority_path(profile, tuple_id=tuple_id)
    script_path = record_path.with_name(f"{tuple_id}.smoke.sbatch")
    authority_script_path = authority_path.with_name(f"{tuple_id}.smoke.sbatch")
    submitted_at = _format_timestamp(_coerce_utc(now))
    record_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        render_preprocessing_runtime_qualification_script(
            profile=profile,
            qualification_tuple=qualification_tuple,
            tuple_id=tuple_id,
            record_path=authority_path,
        )
    )
    submitted = PreprocessingRuntimeQualificationRecord(
        status="submitted",
        tuple_id=tuple_id,
        qualification_tuple=qualification_tuple,
        submitted_at=submitted_at,
        qualified_at=None,
        expires_at=None,
        job_id=None,
        smoke_evidence=None,
    )
    # The scheduled smoke takes the same tuple-addressed lock. Holding it from
    # intent publication through job-id publication closes the race where a
    # very fast smoke could qualify the record before Control rewrites it.
    lock_path = Path(f"{record_path}.lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        _write_record_atomic(record_path, submitted)
        # The tuple-addressed script/job name is the durable correlation before
        # sbatch can return a job id. If Control crashes after acceptance, the
        # job can still publish its own SLURM_JOB_ID; pending intent never qualifies.
        transport = RemoteSlurmTransport(kind=profile.transport, ssh_target=profile.ssh_target, runner=runner)
        if profile.transport == "ssh":
            record_sha256 = _file_sha256(record_path)
            record_staging_token = f"preprocessing-qualification-{record_sha256[:16]}"
            transport.replace_artifact_atomically(
                record_path,
                str(authority_path),
                expected_sha256=record_sha256,
                staging_token=record_staging_token,
                lock_path=f"{authority_path}.lock",
            )
            transport.stage_immutable_artifact(
                script_path,
                str(authority_script_path),
                expected_sha256=_file_sha256(script_path),
                staging_token=f"preprocessing-qualification-{tuple_id[:16]}",
            )
            log_dir = authority_path.parent / "slurm-logs"
            log_dir_result = transport.command(("mkdir", "-p", str(log_dir)))
            if log_dir_result.returncode != 0:
                detail = log_dir_result.stderr.strip() or log_dir_result.stdout.strip() or "mkdir failed"
                raise PreprocessingRuntimeQualificationError(
                    f"failed to create preprocessing qualification Slurm log directory {log_dir}: {detail}"
                )
            submission = transport.submit_script(authority_script_path)
        else:
            submission = transport.submit_script(script_path)
        current = preprocessing_runtime_qualification_record_from_mapping(json.loads(record_path.read_text()))
        if current.status == "qualified":
            return current
        submitted = PreprocessingRuntimeQualificationRecord(
            status="submitted",
            tuple_id=tuple_id,
            qualification_tuple=qualification_tuple,
            submitted_at=submitted_at,
            qualified_at=None,
            expires_at=None,
            job_id=submission.job_id,
            smoke_evidence=None,
        )
        _write_record_atomic(record_path, submitted)
        return submitted


def check_preprocessing_runtime_qualification(
    *,
    profile: ResolvedClusterProfile,
    source_repo: Path,
    now: datetime | None = None,
) -> QualifiedPreprocessingRuntimeSelection:
    """Return the exact current qualified selection or fail closed."""
    expected = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(expected)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    record = _read_qualified_record(record_path, expected=expected, now=now)
    assert record.qualified_at is not None
    assert record.expires_at is not None
    return QualifiedPreprocessingRuntimeSelection(
        qualification_tuple=record.qualification_tuple,
        qualification_record_path=str(record_path),
        qualified_at=record.qualified_at,
        expires_at=record.expires_at,
    )


def resolve_preprocessing_runtime_qualification(
    *,
    profile_name: str,
    config_path: Path,
    source_repo: Path,
    now: datetime | None = None,
    runner: CommandRunner = default_command_runner,
) -> PreprocessingRuntimeQualificationRecord:
    """Resolve qualified cluster authority into the workstation Control root."""
    profile = resolve_cluster_profile(profile_name, config_path=config_path)
    expected = preprocessing_runtime_qualification_tuple(profile, source_repo=source_repo)
    tuple_id = preprocessing_runtime_tuple_id(expected)
    record_path = preprocessing_runtime_qualification_path(profile, tuple_id=tuple_id)
    authority_path = _preprocessing_runtime_qualification_authority_path(profile, tuple_id=tuple_id)
    if profile.transport == "ssh":
        transport = RemoteSlurmTransport(kind=profile.transport, ssh_target=profile.ssh_target, runner=runner)
        try:
            raw = transport.read_immutable_text_artifact_no_follow(str(authority_path), max_bytes=1024 * 1024)
            payload = json.loads(raw)
            record = preprocessing_runtime_qualification_record_from_mapping(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PreprocessingRuntimeQualificationError(
                f"preprocessing runtime qualification authority is missing or invalid: {authority_path}: {exc}"
            ) from exc
        canonical = (json.dumps(record.to_mapping(), indent=2, sort_keys=True) + "\n").encode()
        if raw != canonical:
            raise PreprocessingRuntimeQualificationError(
                f"preprocessing runtime qualification authority is not canonical: {authority_path}"
            )
        _validate_qualified_record(record, expected=expected, record_path=authority_path, now=now)
        _write_record_atomic(record_path, record)
        return record
    return _read_qualified_record(record_path, expected=expected, now=now)


def _read_qualified_record(
    record_path: Path,
    *,
    expected: PreprocessingRuntimeQualificationTuple,
    now: datetime | None,
) -> PreprocessingRuntimeQualificationRecord:
    try:
        payload = json.loads(record_path.read_text())
        record = preprocessing_runtime_qualification_record_from_mapping(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PreprocessingRuntimeQualificationError(
            f"preprocessing runtime qualification is missing or invalid: {record_path}: {exc}"
        ) from exc
    _validate_qualified_record(record, expected=expected, record_path=record_path, now=now)
    return record


def _validate_qualified_record(
    record: PreprocessingRuntimeQualificationRecord,
    *,
    expected: PreprocessingRuntimeQualificationTuple,
    record_path: Path,
    now: datetime | None,
) -> None:
    if record.status != "qualified":
        raise PreprocessingRuntimeQualificationError(
            f"preprocessing runtime qualification is {record.status}, not qualified: {record_path}"
        )
    if record.qualification_tuple != expected:
        raise PreprocessingRuntimeQualificationError("preprocessing runtime qualification tuple does not match")
    assert record.qualified_at is not None
    assert record.expires_at is not None
    if _parse_timestamp(record.expires_at) <= _coerce_utc(now):
        raise PreprocessingRuntimeQualificationError(f"preprocessing runtime qualification expired: {record_path}")


def preprocessing_runtime_qualification_tuple(
    profile: ResolvedClusterProfile,
    *,
    source_repo: Path,
) -> PreprocessingRuntimeQualificationTuple:
    """Resolve the expected tuple without scheduler or filesystem mutation."""
    runtime = profile.preprocessing_runtime
    if runtime is None:
        raise PreprocessingRuntimeQualificationError(
            f"Cluster Profile {profile.name!r} does not define preprocessing_runtime"
        )
    if profile.source_bundle_root is None:
        raise PreprocessingRuntimeQualificationError("preprocessing qualification requires source_bundle_root")
    if profile.runtime_qualification_root is None:
        raise PreprocessingRuntimeQualificationError("preprocessing qualification requires runtime_qualification_root")
    resource = profile.resources.get(_SCHEDULING_CLASS)
    if resource is None:
        raise PreprocessingRuntimeQualificationError(
            f"Cluster Profile {profile.name!r} requires resources.{_SCHEDULING_CLASS}"
        )
    gpu_worker_gres = derive_smoke_gres(
        gres=resource.gres,
        nodes=resource.nodes,
        gpus_per_task=resource.gpus_per_task,
        profile_name=profile.name,
    )
    source_commit = _clean_source_commit(source_repo)
    if source_commit != runtime.source_commit:
        raise PreprocessingRuntimeQualificationError(
            "preprocessing runtime image source commit does not match the selected clean source"
        )
    source_bundle_id = f"bspp-orchestration-{source_commit}"
    source_bundle_path = str(Path(profile.source_bundle_root) / f"{source_bundle_id}.tar.zst")
    return PreprocessingRuntimeQualificationTuple(
        cluster_profile=profile.name,
        scheduling_class=_SCHEDULING_CLASS,
        gpu_worker_gres=gpu_worker_gres,
        cluster_image_path=runtime.cluster_image_path,
        cluster_image_sha256=runtime.cluster_image_sha256,
        oci_digest=runtime.oci_digest,
        source_bundle_id=source_bundle_id,
        source_bundle_path=source_bundle_path,
        source_bundle_sha256=runtime.source_bundle_sha256,
        runtime_contract_id=PREPROCESSING_RUNTIME_CONTRACT_ID,
        adapter_version=PREPROCESSING_ADAPTER_VERSION,
        image_identity=PreprocessingRuntimeImageIdentity(
            source_commit=runtime.source_commit,
            image_lock_sha256=runtime.image_lock_sha256,
            contract_wheel_sha256=runtime.contract_wheel_sha256,
            runtime_wheel_sha256=runtime.runtime_wheel_sha256,
            control_wheel_sha256=runtime.control_wheel_sha256,
            colabfold_version=runtime.colabfold_version,
            mmseqs_version=runtime.mmseqs_version,
            rsync_version=runtime.rsync_version,
            cuda_version=runtime.cuda_version,
        ),
        nodelist=resource.nodelist,
    )


def preprocessing_runtime_qualification_path(
    profile: ResolvedClusterProfile,
    *,
    tuple_id: str,
) -> Path:
    root = profile.runtime_qualification_control_root
    if root is None:
        raise PreprocessingRuntimeQualificationError(
            "preprocessing qualification requires runtime_qualification_control_root"
        )
    return Path(root) / "preprocessing" / profile.name / f"{tuple_id}.json"


def _preprocessing_runtime_qualification_authority_path(
    profile: ResolvedClusterProfile,
    *,
    tuple_id: str,
) -> Path:
    root = profile.runtime_qualification_root
    if root is None:
        raise PreprocessingRuntimeQualificationError("preprocessing qualification requires runtime_qualification_root")
    return Path(root) / "preprocessing" / profile.name / f"{tuple_id}.json"


def render_preprocessing_runtime_qualification_script(
    *,
    profile: ResolvedClusterProfile,
    qualification_tuple: PreprocessingRuntimeQualificationTuple,
    tuple_id: str,
    record_path: Path,
) -> str:
    """Render the scheduled smoke that alone can publish qualified evidence."""
    resource = profile.resources.get(_SCHEDULING_CLASS)
    if resource is None:
        raise PreprocessingRuntimeQualificationError(
            f"Cluster Profile {profile.name!r} requires resources.{_SCHEDULING_CLASS}"
        )
    if tuple_id != preprocessing_runtime_tuple_id(qualification_tuple):
        raise PreprocessingRuntimeQualificationError("preprocessing qualification tuple id does not match")
    if resource.nodelist != qualification_tuple.nodelist:
        raise PreprocessingRuntimeQualificationError("preprocessing qualification nodelist does not match profile")
    expires_hours = profile.runtime_qualification_expires_hours
    source_bundle = Path(qualification_tuple.source_bundle_path)
    mounts = [
        (str(source_bundle.parent), str(source_bundle.parent)),
        (str(record_path.parent), str(record_path.parent)),
    ]
    mounts.extend((mount.source, mount.target) for mount in profile.extra_mounts)
    smoke_database_target = str(PurePosixPath(SELECTED_DATABASE_ROOT).parent)
    mounts = list(_dedupe_mounts(mounts))
    _validate_qualification_mounts(mounts, smoke_database_target=smoke_database_target)
    base_mount_arg = ",".join(f"{source}:{target}" for source, target in mounts)
    tuple_json = json.dumps(qualification_tuple.to_mapping(), sort_keys=True, separators=(",", ":"))
    srun = (
        "srun",
        f"--container-image={qualification_tuple.cluster_image_path}",
        "--no-container-mount-home",
        "/usr/local/bin/entrypoint.sh",
        "/usr/bin/flock",
        "-x",
        f"{record_path}.lock",
        _IMAGE_SMOKE,
        "--qualification-record",
        str(record_path),
    )
    rendered_srun = [shlex.quote(item) for item in srun]
    rendered_srun.insert(2, '--container-mounts="$CONTAINER_MOUNTS"')
    log_dir = record_path.parent / "slurm-logs"
    lines = [
        "#!/usr/bin/env bash",
        "# BSPP preprocessing Runtime Qualification",
        f"# Tuple ID: {tuple_id}",
        f"#SBATCH --job-name=bspp_preprocess_qual_{tuple_id[:12]}",
        f"#SBATCH --partition={resource.partition}",
        f"#SBATCH --account={profile.account}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resource.cpus_per_task}",
        f"#SBATCH --mem={resource.memory}",
        f"#SBATCH --time={resource.time}",
        f"#SBATCH --output={log_dir / f'{tuple_id}.%j.out'}",
        f"#SBATCH --error={log_dir / f'{tuple_id}.%j.err'}",
    ]
    if qualification_tuple.gpu_worker_gres:
        lines.append(f"#SBATCH --gres={qualification_tuple.gpu_worker_gres}")
    if qualification_tuple.nodelist:
        lines.append(f"#SBATCH --nodelist={qualification_tuple.nodelist}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"IMAGE={shlex.quote(qualification_tuple.cluster_image_path)}",
            f"EXPECTED_IMAGE_SHA256={shlex.quote(qualification_tuple.cluster_image_sha256)}",
            '[[ -f "$IMAGE" ]] || { echo "missing preprocessing runtime image: $IMAGE" >&2; exit 127; }',
            'ACTUAL_IMAGE_SHA256="$(sha256sum "$IMAGE" | awk \'{print $1}\')"',
            '[[ "$ACTUAL_IMAGE_SHA256" == "$EXPECTED_IMAGE_SHA256" ]] || { '
            'echo "preprocessing runtime image SHA-256 mismatch" >&2; exit 127; }',
            '[[ -n "${SLURM_JOB_ID:-}" ]] || { '
            'echo "scheduled preprocessing Runtime Qualification requires SLURM_JOB_ID" >&2; exit 127; }',
            "unset BSPP_PREPROCESSING_GPU_EVIDENCE",
            "command -v nvidia-smi >/dev/null || { "
            'echo "nvidia-smi is unavailable on the allocated host" >&2; exit 127; }',
            'BSPP_PREPROCESSING_GPU_EVIDENCE="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)" '
            '|| { echo "host nvidia-smi query failed" >&2; exit 127; }',
            '[[ -n "${BSPP_PREPROCESSING_GPU_EVIDENCE//[[:space:]]/}" ]] || { '
            'echo "host nvidia-smi produced no GPU evidence" >&2; exit 127; }',
            "export BSPP_PREPROCESSING_GPU_EVIDENCE",
            'SMOKE_TMP_PARENT="${SLURM_TMPDIR:-/tmp}"',
            '[[ -d "$SMOKE_TMP_PARENT" && -w "$SMOKE_TMP_PARENT" && ! -L "$SMOKE_TMP_PARENT" ]] || { '
            'echo "scheduled preprocessing Runtime Qualification requires a writable real temporary directory" '
            ">&2; exit 127; }",
            'SMOKE_DATABASE_HOST_ROOT="$(mktemp -d '
            '"$SMOKE_TMP_PARENT/bspp-preprocessing-qualification.${SLURM_JOB_ID}.XXXXXX")"',
            '[[ -d "$SMOKE_DATABASE_HOST_ROOT" && ! -L "$SMOKE_DATABASE_HOST_ROOT" ]] || { '
            'echo "failed to create isolated smoke database root" >&2; exit 127; }',
            "cleanup_smoke_database() {",
            '  find "$SMOKE_DATABASE_HOST_ROOT" -depth -mindepth 1 -delete',
            '  rmdir "$SMOKE_DATABASE_HOST_ROOT"',
            "}",
            "trap cleanup_smoke_database EXIT",
            f"BASE_CONTAINER_MOUNTS={shlex.quote(base_mount_arg)}",
            f"SMOKE_DATABASE_TARGET={shlex.quote(smoke_database_target)}",
            'CONTAINER_MOUNTS="${BASE_CONTAINER_MOUNTS},${SMOKE_DATABASE_HOST_ROOT}:${SMOKE_DATABASE_TARGET}"',
            f"export BSPP_PREPROCESSING_QUALIFICATION_TUPLE={shlex.quote(tuple_json)}",
            f"export BSPP_PREPROCESSING_QUALIFICATION_TUPLE_ID={shlex.quote(tuple_id)}",
            f"export BSPP_PREPROCESSING_QUALIFICATION_EXPIRES_HOURS={expires_hours}",
            f"export BSPP_PREPROCESSING_SOURCE_BUNDLE={shlex.quote(str(source_bundle))}",
            'export BSPP_PREPROCESSING_IMAGE_SHA256="$ACTUAL_IMAGE_SHA256"',
            (" \\" + "\n  ").join(rendered_srun),
            "",
        ]
    )
    return "\n".join(lines)


def render_preprocessing_runtime_qualification_yaml(
    record: PreprocessingRuntimeQualificationRecord,
    *,
    record_path: Path,
) -> str:
    return yaml.safe_dump(
        {
            "preprocessing_runtime_qualification": {
                "status": record.status,
                "tuple_id": record.tuple_id,
                "record_path": str(record_path),
                "job_id": record.job_id,
                "qualified_at": record.qualified_at,
                "expires_at": record.expires_at,
            }
        },
        sort_keys=False,
    )


def _clean_source_commit(source_repo: Path) -> str:
    source_repo = source_repo.resolve()
    commit = _git(source_repo, "rev-parse", "HEAD")
    status = _git(source_repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise PreprocessingRuntimeQualificationError("preprocessing Runtime Qualification requires clean source")
    return commit


def _git(source_repo: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(source_repo), *args), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise PreprocessingRuntimeQualificationError(result.stderr.strip() or "git source identity failed")
    return result.stdout.strip()


def _write_record_atomic(path: Path, record: PreprocessingRuntimeQualificationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record.to_mapping(), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _dedupe_mounts(mounts: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(dict.fromkeys(mounts))


def _validate_qualification_mounts(
    mounts: list[tuple[str, str]],
    *,
    smoke_database_target: str,
) -> None:
    protected = PurePosixPath(smoke_database_target)
    for source, target in mounts:
        for endpoint in (source, target):
            path = PurePosixPath(endpoint)
            if not path.is_absolute() or ".." in path.parts or str(path) != endpoint:
                raise PreprocessingRuntimeQualificationError(
                    "preprocessing qualification mount endpoints must be absolute normalized paths"
                )
            if _paths_overlap(path, protected):
                raise PreprocessingRuntimeQualificationError(
                    "preprocessing qualification mount overlaps protected smoke database namespace"
                )


def _paths_overlap(first: PurePosixPath, second: PurePosixPath) -> bool:
    return first == second or first in second.parents or second in first.parents


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        raise ValueError("preprocessing qualification clock must be timezone-aware")
    return value.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


__all__ = [
    "PreprocessingRuntimeQualificationError",
    "check_preprocessing_runtime_qualification",
    "derive_smoke_gres",
    "preprocessing_runtime_qualification_path",
    "preprocessing_runtime_qualification_tuple",
    "qualify_preprocessing_runtime",
    "render_preprocessing_runtime_qualification_script",
    "render_preprocessing_runtime_qualification_yaml",
    "resolve_preprocessing_runtime_qualification",
]
