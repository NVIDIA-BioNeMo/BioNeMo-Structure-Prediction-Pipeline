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

"""Runtime Qualification records for Control Plane submissions."""

from __future__ import annotations

import errno
import json
import os
import re
import shlex
import stat
import subprocess
import tarfile
import tempfile
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import yaml

from bspp.orchestration.contract.runspec import BAKED_TOOLKIT_COMMIT
from bspp.orchestration.contract.runtime_qualification import (
    RUNTIME_IPSAE_EXPECTED_SCORE_AB,
    RUNTIME_IPSAE_EXPECTED_SCORE_BA,
    RUNTIME_IPSAE_FIXTURE_MODEL_ID,
    RUNTIME_IPSAE_FIXTURE_PAE,
    RUNTIME_IPSAE_FIXTURE_PDB,
    RuntimeQualificationCheck,
    RuntimeQualificationRecord,
    RuntimeQualificationSnapshot,
    runtime_ipsae_evidence_from_mapping,
    runtime_qualification_payload_from_mapping,
)
from bspp.orchestration.contract.source_package import (
    MANIFEST_MEMBER,
    SourcePackageIdentity,
    build_source_package,
    source_package_identity_from_mapping,
    verify_source_package,
)
from bspp.orchestration.contract.versioning import UnsupportedSchemaVersionError
from bspp.orchestration.control.execution_bootstrap import (
    PIXI_PYTHON_PATH,
    ImageIdentity,
    identify_runtime_image,
    image_identity_from_mapping,
    render_governed_srun,
)
from bspp.orchestration.control.profiles import ResolvedClusterProfile, resolve_cluster_profile
from bspp.orchestration.control.runtime_qualification_validation import (
    ValidatedRuntimeQualification,
    expected_runtime_qualification_attempt_paths,
    validate_promoted_runtime_qualification,
    validate_runtime_qualification_attempt_binding,
)
from bspp.orchestration.control.transport import (
    CommandRunner,
    RemoteSlurmTransport,
    command_argv,
    default_command_runner,
)
from bspp.orchestration.control.workflow_rendering import (
    BAKED_TOOLKIT_CONTAINER_ROOT,
    TOOLKIT_CONTAINER_ROOT,
)

_SCHEDULING_CLASS = "gpu_worker"
_SOURCE_PACKAGE_CONTAINER = "/run/bspp/source-package.tar"
_TOOLKIT_PACKAGE_CONTAINER = "/run/bspp/toolkit-package.tar"
_QUALIFICATION_CONTAINER = "/run/bspp/runtime-qualification.json"
_MAX_QUALIFICATION_RESULT_BYTES = 1024 * 1024
_MAX_PUBLICATION_COMPATIBILITY_BYTES = 4096
_QUALIFICATION_SNAPSHOT_CAPTURE_ATTEMPTS = 3
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REMOTE_RESULT_PROBE = """import hashlib,json,os,stat,sys
p=sys.argv[1]; limit=int(sys.argv[2]); before=os.lstat(p)
if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): raise SystemExit('unsafe result')
if before.st_size>limit: raise SystemExit('oversize result')
fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
try:
 opened=os.fstat(fd); h=hashlib.sha256()
 while chunk:=os.read(fd,65536): h.update(chunk)
 after=os.fstat(fd)
finally: os.close(fd)
current=os.lstat(p)
sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
if sig(before)!=sig(opened) or sig(opened)!=sig(after) or sig(after)!=sig(current): raise SystemExit('changed result')
print(json.dumps({'size_bytes':opened.st_size,'sha256':h.hexdigest()}))
"""
_REMOTE_ATTEMPT_CLEANUP = """import os,re,shutil,stat,sys
path,configured_root,profile,tuple_id,token=sys.argv[1:]
if not re.fullmatch(r'[A-Za-z0-9_.-]+',profile): raise SystemExit('unsafe attempt cleanup profile')
root=os.path.join(configured_root,profile,'attempts',tuple_id)
expected=os.path.join(root,token)
values=(configured_root,root,expected,path)
if any(not os.path.isabs(p) or os.path.normpath(p)!=p or os.path.realpath(p)!=p for p in values):
 raise SystemExit('unsafe attempt cleanup ancestry')
if not re.fullmatch(r'[0-9a-f]{64}',tuple_id) or os.path.basename(root)!=tuple_id:
 raise SystemExit('unsafe attempt cleanup tuple')
if path!=expected or not re.fullmatch(r'[0-9a-f]{32}',token) or os.path.basename(path)!=token:
 raise SystemExit('unsafe attempt cleanup target')
try: info=os.lstat(path)
except FileNotFoundError: raise SystemExit(0)
if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise SystemExit('unsafe attempt cleanup')
shutil.rmtree(path)
"""


class RuntimeQualificationConfigError(ValueError):
    """Raised when the Cluster Profile lacks qualification configuration."""


class RuntimeQualificationSourceError(ValueError):
    """Raised when the source checkout cannot identify a qualification tuple."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def _read_bounded_stable_bytes(
    path: Path,
    *,
    expected_size_bytes: int | None = None,
) -> tuple[bytes, str]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise ValueError("qualification result path is unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("qualification result path is unsafe")
        if before.st_size > _MAX_QUALIFICATION_RESULT_BYTES:
            raise ValueError("qualification result is oversize")
        if expected_size_bytes is not None and before.st_size != expected_size_bytes:
            raise ValueError("qualification result size differs from the resolved snapshot")
        chunks: list[bytes] = []
        digest = sha256()
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                raise ValueError("qualification result was truncated")
            chunks.append(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("qualification result grew while reading")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)

        def signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

        if signature(before) != signature(after) or signature(after) != signature(current):
            raise ValueError("qualification result changed while reading")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def verify_runtime_qualification_snapshot_path(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> None:
    """Verify that a mutable pathname still names one exact resolved snapshot."""
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("Runtime Qualification snapshot SHA-256 is invalid")
    if (
        isinstance(expected_size_bytes, bool)
        or not isinstance(expected_size_bytes, int)
        or expected_size_bytes < 0
        or expected_size_bytes > _MAX_QUALIFICATION_RESULT_BYTES
    ):
        raise ValueError("Runtime Qualification snapshot size is invalid")
    try:
        path_info = path.lstat()
    except OSError as exc:
        raise ValueError("Runtime Qualification snapshot path is not a regular file") from exc
    if not stat.S_ISREG(path_info.st_mode):
        raise ValueError("Runtime Qualification snapshot path is not a regular file")
    if path_info.st_size != expected_size_bytes:
        raise ValueError("Runtime Qualification snapshot path size differs from the resolved snapshot")
    try:
        _document, observed_sha256 = _read_bounded_stable_bytes(
            path,
            expected_size_bytes=expected_size_bytes,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("Runtime Qualification snapshot path changed or is not a safe regular file") from exc
    if observed_sha256 != expected_sha256:
        raise ValueError("Runtime Qualification snapshot path SHA-256 differs from the resolved snapshot")


def _read_bounded_stable_json(path: Path) -> dict[str, object]:
    raw, _digest = _read_bounded_stable_bytes(path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("qualification result is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("qualification result must be a mapping")
    return value


def _capture_runtime_qualification_payload(path: Path) -> dict[str, object]:
    """Read a settled authority mapping across a bounded replacement window."""
    last_error: OSError | ValueError | None = None
    for _attempt in range(_QUALIFICATION_SNAPSHOT_CAPTURE_ATTEMPTS):
        try:
            return runtime_qualification_payload_from_mapping(_read_bounded_stable_json(path))
        except (OSError, ValueError) as exc:
            last_error = exc
    if last_error is None:  # pragma: no cover - the positive attempt count is fixed above
        raise ValueError("Runtime Qualification authority capture has no configured attempts")
    raise last_error


def qualify_runtime(
    *,
    profile_name: str,
    config_path: Path,
    source_repo: Path,
    now: datetime | None = None,
    runner: CommandRunner = default_command_runner,
    source_package_identity_path: Path | None = None,
) -> RuntimeQualificationRecord:
    """Submit a scheduled smoke job that can qualify a concrete runtime tuple."""
    profile = resolve_cluster_profile(profile_name, config_path=config_path)
    qualification_tuple = _qualification_tuple(
        profile, source_repo=source_repo, source_package_identity_path=source_package_identity_path, runner=runner
    )
    tuple_id = _tuple_id(qualification_tuple)
    remote_root = _require_profile_value(profile.runtime_qualification_root, "runtime_qualification_root")
    control_root = _require_profile_value(
        profile.runtime_qualification_control_root, "runtime_qualification_control_root"
    )
    expires_hours = _require_profile_value(
        profile.runtime_qualification_expires_hours,
        "runtime_qualification_expires_hours",
    )
    submitted_at = _coerce_utc(now)
    path = _record_path(control_root, profile_name=profile.name, tuple_id=tuple_id)
    attempt_history: list[object] = []
    prepared_retry: Mapping[str, object] | None = None
    if path.exists():
        try:
            existing = runtime_qualification_payload_from_mapping(json.loads(path.read_text()))
        except (OSError, ValueError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            previous_history = existing.get("attempt_history")
            if isinstance(previous_history, list):
                attempt_history.extend(previous_history)
            if (
                existing.get("tuple_id") == tuple_id
                and existing.get("tuple") == qualification_tuple
                and existing.get("status") in {"prepared", "submitting", "submitted"}
            ):
                _validate_persisted_attempt_binding(
                    existing,
                    profile=profile,
                    tuple_id=tuple_id,
                    control_root=control_root,
                    remote_root=remote_root,
                    record_path=path,
                )
        reusable_status = existing.get("status") if isinstance(existing, dict) else None
        if reusable_status == "qualified" and isinstance(existing, dict):
            try:
                if submitted_at >= _parse_instant(existing.get("expires_at")):
                    reusable_status = "expired"
            except ValueError:
                reusable_status = "invalid"
        if reusable_status == "submitted" and isinstance(existing, dict):
            smoke_job_value = existing.get("smoke_job")
            bound_job = smoke_job_value.get("job_id") if isinstance(smoke_job_value, dict) else None
            if not isinstance(bound_job, str) or not bound_job:
                reusable_status = "invalid"
            else:
                states = RemoteSlurmTransport(
                    kind=profile.transport, ssh_target=profile.ssh_target, runner=runner
                ).query_job_states((bound_job,))
                if not states:
                    raise ValueError(
                        "ambiguous Runtime Qualification scheduler accounting for submitted job; refusing resubmit"
                    )
                if states[0].state in {
                    "BOOT_FAIL",
                    "CANCELLED",
                    "DEADLINE",
                    "FAILED",
                    "NODE_FAIL",
                    "OUT_OF_MEMORY",
                    "PREEMPTED",
                    "TIMEOUT",
                }:
                    reusable_status = "terminal-or-lost"
        if (
            isinstance(existing, dict)
            and existing.get("tuple_id") == tuple_id
            and existing.get("tuple") == qualification_tuple
            and reusable_status in {"submitted", "qualified"}
        ):
            smoke_job = existing.get("smoke_job")
            job_id = smoke_job.get("job_id") if isinstance(smoke_job, dict) else None
            script_value = smoke_job.get("script_path") if isinstance(smoke_job, dict) else None
            return RuntimeQualificationRecord(
                tuple_id=tuple_id,
                path=path,
                status=str(existing["status"]),
                job_id=job_id if isinstance(job_id, str) else None,
                script_path=Path(script_value) if isinstance(script_value, str) else None,
            )
        if (
            isinstance(existing, dict)
            and existing.get("tuple_id") == tuple_id
            and existing.get("tuple") == qualification_tuple
            and existing.get("status") == "submitting"
        ):
            smoke_job = existing.get("smoke_job")
            token = smoke_job.get("attempt_token") if isinstance(smoke_job, dict) else None
            if not isinstance(token, str):
                raise ValueError("submitting Runtime Qualification record lacks attempt token")
            transport = RemoteSlurmTransport(kind=profile.transport, ssh_target=profile.ssh_target, runner=runner)
            reconciled_job = transport.find_runtime_qualification_job(
                attempt_token=token, profile=profile.name, owner=profile.owner
            )
            if reconciled_job is not None:
                reconciled = dict(existing)
                reconciled_job_payload = dict(smoke_job) if isinstance(smoke_job, dict) else {}
                reconciled_job_payload["job_id"] = reconciled_job
                reconciled_job_payload["submit_command"] = ["scheduler-reconciliation"]
                reconciled["status"] = "submitted"
                reconciled["smoke_job"] = reconciled_job_payload
                _write_json_atomic(path, reconciled)
                script_value = reconciled_job_payload.get("script_path")
                return RuntimeQualificationRecord(
                    tuple_id=tuple_id,
                    path=path,
                    status="submitted",
                    job_id=reconciled_job,
                    script_path=Path(script_value) if isinstance(script_value, str) else None,
                )
            raise ValueError("ambiguous Runtime Qualification submission; scheduler accounting has not settled")
        if (
            isinstance(existing, dict)
            and existing.get("tuple_id") == tuple_id
            and existing.get("tuple") == qualification_tuple
            and existing.get("status") == "prepared"
            and isinstance(existing.get("smoke_job"), dict)
        ):
            prepared_value = existing.get("smoke_job")
            assert isinstance(prepared_value, dict)
            prepared_retry = prepared_value
        if isinstance(existing, dict) and isinstance(existing.get("smoke_job"), dict):
            attempt_history.append({"status": existing.get("status"), "smoke_job": existing["smoke_job"]})
    retry_token = prepared_retry.get("attempt_token") if prepared_retry is not None else None
    if isinstance(retry_token, str) and (
        len(retry_token) != 32 or any(character not in "0123456789abcdef" for character in retry_token)
    ):
        raise ValueError("prepared Runtime Qualification attempt token is invalid")
    attempt_token = retry_token if isinstance(retry_token, str) else uuid.uuid4().hex
    if prepared_retry is not None:
        required_retry_paths = {
            name: prepared_retry.get(name)
            for name in (
                "script_path",
                "input_path",
                "result_path",
                "remote_script_path",
                "remote_input_path",
                "remote_result_path",
            )
        }
        if not all(isinstance(value, str) for value in required_retry_paths.values()):
            raise ValueError("prepared Runtime Qualification record lacks persisted attempt paths")
        script_path = Path(str(required_retry_paths["script_path"]))
        input_path = Path(str(required_retry_paths["input_path"]))
        result_path = Path(str(required_retry_paths["result_path"]))
        remote_script_path = Path(str(required_retry_paths["remote_script_path"]))
        remote_input_path = Path(str(required_retry_paths["remote_input_path"]))
        remote_result_path = Path(str(required_retry_paths["remote_result_path"]))
        expected_paths = _expected_attempt_paths(
            profile=profile,
            tuple_id=tuple_id,
            attempt_token=attempt_token,
            control_root=control_root,
            remote_root=remote_root,
        )
        observed_paths = {
            "script_path": script_path,
            "input_path": input_path,
            "result_path": result_path,
            "remote_script_path": remote_script_path,
            "remote_input_path": remote_input_path,
            "remote_result_path": remote_result_path,
        }
        if observed_paths != expected_paths:
            raise ValueError("prepared Runtime Qualification paths do not match configured attempt ancestry")
    else:
        attempt_paths = _expected_attempt_paths(
            profile=profile,
            tuple_id=tuple_id,
            attempt_token=attempt_token,
            control_root=control_root,
            remote_root=remote_root,
        )
        script_path = attempt_paths["script_path"]
        input_path = attempt_paths["input_path"]
        result_path = attempt_paths["result_path"]
        remote_script_path = attempt_paths["remote_script_path"]
        remote_input_path = attempt_paths["remote_input_path"]
        remote_result_path = attempt_paths["remote_result_path"]
    script_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if profile.transport != "ssh":
        (path.parent / "slurm-logs").mkdir(parents=True, exist_ok=True)
    script_path.write_text(
        render_runtime_qualification_smoke_script(
            profile=profile,
            qualification_tuple=qualification_tuple,
            tuple_id=tuple_id,
            record_path=remote_input_path,
            result_path=remote_result_path,
            attempt_token=attempt_token,
            expires_hours=expires_hours,
        )
    )
    pending_payload = _submitted_payload(
        profile=profile,
        qualification_tuple=qualification_tuple,
        tuple_id=tuple_id,
        record_path=path,
        script_path=script_path,
        submitted_at=submitted_at,
        expires_hours=expires_hours,
        job_id=None,
        submit_command=(),
        status="prepared",
        attempt_token=attempt_token,
        input_path=input_path,
        result_path=result_path,
        remote_script_path=remote_script_path,
        remote_input_path=remote_input_path,
        remote_result_path=remote_result_path,
        attempt_history=attempt_history,
    )
    _write_json_atomic(path, pending_payload)
    if input_path != path:
        _write_json_atomic(input_path, pending_payload)
    transport = RemoteSlurmTransport(kind=profile.transport, ssh_target=profile.ssh_target, runner=runner)
    if profile.transport == "ssh":
        try:
            hierarchy_result = transport.command(("mkdir", "-m", "700", "-p", str(remote_script_path.parent.parent)))
            if hierarchy_result.returncode != 0:
                raise ValueError(hierarchy_result.stderr.strip() or "remote qualification hierarchy mkdir failed")
            mkdir_argv = (
                ("mkdir", "-m", "700", "-p", str(remote_script_path.parent))
                if prepared_retry is not None
                else ("mkdir", "-m", "700", str(remote_script_path.parent))
            )
            mkdir_result = transport.command(mkdir_argv)
            if mkdir_result.returncode != 0:
                raise ValueError(mkdir_result.stderr.strip() or "remote qualification attempt mkdir failed")
            for local, remote in ((input_path, remote_input_path), (script_path, remote_script_path)):
                copied = transport.copy_artifact(local, str(remote))
                if copied.returncode != 0:
                    raise ValueError(copied.stderr.strip() or "remote qualification attempt staging failed")
        except ValueError as staging_error:
            cleanup = transport.command(
                (
                    "/usr/bin/python3",
                    "-I",
                    "-S",
                    "-c",
                    _REMOTE_ATTEMPT_CLEANUP,
                    str(remote_script_path.parent),
                    remote_root,
                    profile.name,
                    tuple_id,
                    attempt_token,
                )
            )
            if cleanup.returncode != 0:
                raise ValueError(
                    cleanup.stderr.strip() or "remote qualification attempt cleanup failed"
                ) from staging_error
            raise
    submitting_payload = _submitted_payload(
        profile=profile,
        qualification_tuple=qualification_tuple,
        tuple_id=tuple_id,
        record_path=path,
        script_path=script_path,
        submitted_at=submitted_at,
        expires_hours=expires_hours,
        job_id=None,
        submit_command=(),
        status="submitting",
        attempt_token=attempt_token,
        input_path=input_path,
        result_path=result_path,
        remote_script_path=remote_script_path,
        remote_input_path=remote_input_path,
        remote_result_path=remote_result_path,
        attempt_history=attempt_history,
    )
    _write_json_atomic(path, submitting_payload)
    submission = transport.submit_script(remote_script_path)
    current_payload = runtime_qualification_payload_from_mapping(json.loads(path.read_text()))
    if current_payload.get("status") != "qualified":
        submitted_payload = _submitted_payload(
            profile=profile,
            qualification_tuple=qualification_tuple,
            tuple_id=tuple_id,
            record_path=path,
            script_path=script_path,
            submitted_at=submitted_at,
            expires_hours=expires_hours,
            job_id=submission.job_id,
            submit_command=submission.command,
            status="submitted",
            attempt_token=attempt_token,
            input_path=input_path,
            result_path=result_path,
            remote_script_path=remote_script_path,
            remote_input_path=remote_input_path,
            remote_result_path=remote_result_path,
            attempt_history=attempt_history,
        )
        _write_json_atomic(path, submitted_payload)
    return RuntimeQualificationRecord(
        tuple_id=tuple_id,
        path=path,
        status="submitted",
        job_id=submission.job_id,
        script_path=script_path,
    )


def check_runtime_qualification(
    *,
    profile_name: str,
    config_path: Path,
    source_repo: Path,
    now: datetime | None = None,
    runner: CommandRunner = default_command_runner,
) -> RuntimeQualificationCheck:
    """Check whether the current runtime tuple has unexpired qualification evidence."""
    profile = resolve_cluster_profile(profile_name, config_path=config_path)
    try:
        discovered = _discover_source_package_identity(profile, source_repo=source_repo)
        qualification_tuple = _qualification_tuple(
            profile, source_repo=source_repo, source_package_identity=discovered, runner=runner
        )
        tuple_id = _tuple_id(qualification_tuple)
        root = _require_profile_value(
            profile.runtime_qualification_control_root,
            "runtime_qualification_control_root",
        )
    except RuntimeQualificationConfigError:
        return RuntimeQualificationCheck(
            current=False,
            reason="unconfigured",
            tuple_id=None,
            record_path=Path(profile.runtime_qualification_control_root or "."),
        )
    except RuntimeQualificationSourceError as exc:
        return RuntimeQualificationCheck(
            current=False,
            reason=exc.reason,
            tuple_id=None,
            record_path=Path(profile.runtime_qualification_control_root or "."),
        )
    except ValueError:
        # A non-discoverable tuple (e.g. no staged source package identity for
        # this commit) means the runtime tuple cannot be qualified right now;
        # report not-qualified so dev runs may warn-and-continue.
        return RuntimeQualificationCheck(
            current=False,
            reason="not-qualified",
            tuple_id=None,
            record_path=Path(profile.runtime_qualification_control_root or "."),
        )
    record_path = Path(root) / profile.name / f"{tuple_id}.json"
    try:
        record_path.lstat()
    except FileNotFoundError:
        try:
            reason = "tuple-mismatch" if any((Path(root) / profile.name).glob("*.json")) else "missing"
        except OSError:
            reason = "invalid-record"
        return RuntimeQualificationCheck(
            current=False,
            reason=reason,
            tuple_id=tuple_id,
            record_path=record_path,
        )
    except OSError:
        return RuntimeQualificationCheck(
            current=False,
            reason="invalid-record",
            tuple_id=tuple_id,
            record_path=record_path,
        )
    try:
        payload = _capture_runtime_qualification_payload(record_path)
    except UnsupportedSchemaVersionError:
        return RuntimeQualificationCheck(
            current=False,
            reason="unsupported-schema-version",
            tuple_id=tuple_id,
            record_path=record_path,
        )
    except (OSError, ValueError):
        return RuntimeQualificationCheck(
            current=False,
            reason="invalid-record",
            tuple_id=tuple_id,
            record_path=record_path,
        )
    if payload.get("tuple") != qualification_tuple or payload.get("tuple_id") != tuple_id:
        return RuntimeQualificationCheck(
            current=False,
            reason="tuple-mismatch",
            tuple_id=tuple_id,
            record_path=record_path,
        )
    if payload.get("status") in {"prepared", "submitting", "submitted"}:
        try:
            _validate_persisted_attempt_binding(
                payload,
                profile=profile,
                tuple_id=tuple_id,
                control_root=root,
                remote_root=_require_profile_value(profile.runtime_qualification_root, "runtime_qualification_root"),
                record_path=record_path,
            )
        except ValueError:
            return RuntimeQualificationCheck(
                current=False,
                reason="invalid-record",
                tuple_id=tuple_id,
                record_path=record_path,
            )
    if payload.get("status") == "submitted":
        smoke_job = payload.get("smoke_job")
        smoke_job_mapping = smoke_job if isinstance(smoke_job, dict) else {}
        configured_result = smoke_job_mapping.get("result_path")
        result_path = (
            Path(configured_result)
            if isinstance(configured_result, str)
            else record_path.parent / "results" / tuple_id / "smoke.json"
        )
        remote_result = smoke_job_mapping.get("remote_result_path")
        if profile.transport == "ssh" and isinstance(remote_result, str) and not result_path.exists():
            result_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = result_path.with_name(f".{result_path.name}.{uuid.uuid4().hex}.fetch")
            transport = RemoteSlurmTransport(kind=profile.transport, ssh_target=profile.ssh_target, runner=runner)
            probe = transport.command(
                (
                    "/usr/bin/python3",
                    "-I",
                    "-S",
                    "-c",
                    _REMOTE_RESULT_PROBE,
                    remote_result,
                    str(_MAX_QUALIFICATION_RESULT_BYTES),
                )
            )
            try:
                expected_fetch = json.loads(probe.stdout) if probe.returncode == 0 else None
            except json.JSONDecodeError:
                expected_fetch = None
            fetched = transport.fetch_artifact(remote_result, temporary) if isinstance(expected_fetch, dict) else None
            if fetched is not None and fetched.returncode == 0:
                try:
                    info = temporary.lstat()
                    _raw, observed_sha = _read_bounded_stable_bytes(temporary)
                    if (
                        not temporary.is_symlink()
                        and info.st_size <= _MAX_QUALIFICATION_RESULT_BYTES
                        and expected_fetch == {"size_bytes": info.st_size, "sha256": observed_sha}
                    ):
                        os.replace(temporary, result_path)
                finally:
                    temporary.unlink(missing_ok=True)
        try:
            result = _read_bounded_stable_json(result_path)
        except (OSError, ValueError):
            result = None
        expected_job = smoke_job.get("job_id") if isinstance(smoke_job, dict) else None
        expected_token = smoke_job.get("attempt_token") if isinstance(smoke_job, dict) else None
        scheduler_succeeded = False
        if isinstance(result, dict) and isinstance(expected_job, str):
            try:
                observed_states = RemoteSlurmTransport(
                    kind=profile.transport, ssh_target=profile.ssh_target, runner=runner
                ).query_job_states((expected_job,))
            except ValueError:
                observed_states = ()
            scheduler_succeeded = bool(observed_states and observed_states[0].state == "COMPLETED")
        tuple_selected = qualification_tuple.get("selected_source")
        tuple_source_kind = tuple_selected.get("source_kind") if isinstance(tuple_selected, Mapping) else None
        if (
            isinstance(result, dict)
            and result.get("status") == "succeeded"
            and result.get("tuple_id") == tuple_id
            and result.get("job_id") == expected_job
            and result.get("attempt_token") == expected_token
            and scheduler_succeeded
            and result.get("source_package_identity") == qualification_tuple.get("source_package_identity")
            and _valid_selected_source_identity(
                result.get("selected_source_identity"),
                qualification_tuple.get("selected_source"),
            )
            and (
                tuple_source_kind == "baked"
                or result.get("toolkit_package_identity") == qualification_tuple.get("toolkit_package_identity")
            )
            and result.get("image_identity") == qualification_tuple.get("image_identity")
            and isinstance(result.get("python"), str)
            and bool(result.get("python"))
            and isinstance(result.get("gpu"), str)
            and bool(result.get("gpu"))
            and isinstance((bootstrap_sha256 := result.get("bootstrap_sha256")), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", bootstrap_sha256))
            and _valid_runtime_ipsae_evidence(
                result.get("runtime_ipsae"),
                expected_revision=_toolkit_revision(qualification_tuple),
                source_kind=tuple_source_kind,
            )
            and _runtime_ipsae_artifacts_match(
                result.get("runtime_ipsae"),
                result_path=result_path,
                remote_result_path=remote_result if isinstance(remote_result, str) else None,
                profile=profile,
                runner=runner,
            )
            and _valid_publication_compatibility_evidence(result.get("publication_compatibility"))
            and _publication_compatibility_artifact_matches(
                result.get("publication_compatibility"),
                result_path=result_path,
                remote_result_path=remote_result if isinstance(remote_result, str) else None,
                profile=profile,
                runner=runner,
            )
        ):
            promoted = dict(payload)
            promoted.update(
                {
                    "status": "qualified",
                    "evidence_status": "qualified",
                    "qualified_at": _coerce_utc(now).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "smoke_evidence": result,
                }
            )
            _write_json_atomic(record_path, promoted)
            payload = runtime_qualification_payload_from_mapping(promoted)
    if payload.get("status") != "qualified" or payload.get("evidence_status") != "qualified":
        return RuntimeQualificationCheck(
            current=False,
            reason="not-qualified",
            tuple_id=tuple_id,
            record_path=record_path,
        )
    try:
        snapshot, validated = _capture_validated_qualification_snapshot(
            record_path,
            profile=profile,
            source_repo=source_repo,
            tuple_id=tuple_id,
        )
    except (OSError, ValueError):
        return RuntimeQualificationCheck(
            current=False,
            reason="invalid-record",
            tuple_id=tuple_id,
            record_path=record_path,
        )
    observed_at = _coerce_utc(now)
    qualified_at = _parse_instant(validated.qualified_at)
    expires_at = _parse_instant(validated.expires_at)
    if observed_at < qualified_at:
        return RuntimeQualificationCheck(
            current=False,
            reason="invalid-record",
            tuple_id=tuple_id,
            record_path=record_path,
        )
    if observed_at >= expires_at:
        return RuntimeQualificationCheck(
            current=False,
            reason="stale",
            tuple_id=tuple_id,
            record_path=record_path,
        )
    return RuntimeQualificationCheck(
        current=True,
        reason="current",
        tuple_id=tuple_id,
        record_path=record_path,
        snapshot=snapshot,
    )


def _capture_validated_qualification_snapshot(
    record_path: Path,
    *,
    profile: ResolvedClusterProfile,
    source_repo: Path,
    tuple_id: str,
) -> tuple[RuntimeQualificationSnapshot, ValidatedRuntimeQualification]:
    """Capture the settled canonical authority and authenticate those exact bytes."""
    last_error: OSError | ValueError | None = None
    for _attempt in range(_QUALIFICATION_SNAPSHOT_CAPTURE_ATTEMPTS):
        try:
            document, digest = _read_bounded_stable_bytes(record_path)
        except (OSError, ValueError) as exc:
            last_error = exc
            continue
        validated = validate_promoted_runtime_qualification(
            document,
            profile=profile,
            source_repo=source_repo,
            observed_at=None,
        )
        if validated.tuple_id != tuple_id:
            raise ValueError("Runtime Qualification snapshot tuple differs from the expected tuple")
        return (
            RuntimeQualificationSnapshot(
                document=document,
                sha256=digest,
                size_bytes=len(document),
            ),
            validated,
        )
    raise ValueError("Runtime Qualification authority did not settle during snapshot capture") from last_error


def render_runtime_qualification_smoke_script(
    *,
    profile: ResolvedClusterProfile,
    qualification_tuple: Mapping[str, object],
    tuple_id: str,
    record_path: Path,
    result_path: Path | None = None,
    attempt_token: str | None = None,
    expires_hours: int,
) -> str:
    """Render the scheduled Runtime Qualification smoke sbatch script."""
    resource = profile.resources.get(_SCHEDULING_CLASS)
    if resource is None:
        msg = f"Runtime Qualification requires Cluster Profile resources.{_SCHEDULING_CLASS}"
        raise RuntimeQualificationConfigError(msg)
    source_bundle_path = _tuple_path(qualification_tuple, "source_bundle_path")
    image_path = _tuple_string(qualification_tuple, "execution_runtime_image")
    log_dir = record_path.parent / "slurm-logs"
    image_identity = ImageIdentity(
        format_version=1,
        policy=_tuple_string(qualification_tuple, "runtime_image_policy"),  # type: ignore[arg-type]
        path=Path(image_path),
        size_bytes=_tuple_int(qualification_tuple, "runtime_image_size_bytes"),
        sha256=_tuple_string(qualification_tuple, "runtime_image_sha256"),
    )
    result_path = result_path or record_path.parent / "results" / tuple_id / "smoke.json"
    selected_source = qualification_tuple.get("selected_source")
    source_kind = selected_source.get("source_kind") if isinstance(selected_source, Mapping) else "override"
    if source_kind not in ("baked", "override"):
        source_kind = "override"
    expected_revision = _toolkit_revision(qualification_tuple)
    toolkit_root_template = str(BAKED_TOOLKIT_CONTAINER_ROOT) if source_kind == "baked" else "{BSPP_TOOLKIT_ROOT}"
    qualification_code = _runtime_ipsae_smoke_python()
    compatibility_launcher = _publication_compatibility_launcher()
    compatibility_argv = (
        str(PIXI_PYTHON_PATH),
        "-I",
        "-c",
        compatibility_launcher,
        "{BSPP_SOURCE_ROOT}",
        "bspp.orchestration.runtime.postprocessing.publication_compatibility",
        "--root",
        "/run/bspp/result/publication-compatibility",
        "--output",
        "/run/bspp/result/publication-compatibility.json",
    )
    qualification_argv = (
        str(PIXI_PYTHON_PATH),
        "-I",
        "-S",
        "-c",
        qualification_code,
        f"/run/bspp/result/{result_path.name}",
        tuple_id,
        json.dumps(dict(qualification_tuple), sort_keys=True, separators=(",", ":")),
        attempt_token or "legacy",
        toolkit_root_template,
        source_kind,
        expected_revision or "",
        "nvidia-smi",
        RUNTIME_IPSAE_FIXTURE_MODEL_ID,
        RUNTIME_IPSAE_FIXTURE_PDB,
        RUNTIME_IPSAE_FIXTURE_PAE,
        "-meta_v1.json",
        RUNTIME_IPSAE_EXPECTED_SCORE_AB,
        RUNTIME_IPSAE_EXPECTED_SCORE_BA,
        "/opt/bspp/execution_bootstrap.py",
        "/run/bspp/result/publication-compatibility.json",
    )
    is_baked = source_kind == "baked"
    smoke_mounts = _smoke_container_mounts(
        source_bundle_path=source_bundle_path,
        toolkit_package_path=_tuple_path(qualification_tuple, "toolkit_package_path") if not is_baked else None,
        record_path=record_path,
        result_path=result_path,
    )
    bootstrap_args: tuple[str, ...]
    if is_baked:
        bootstrap_args = (
            "run",
            "--identity-record",
            _QUALIFICATION_CONTAINER,
            "--source-package",
            _SOURCE_PACKAGE_CONTAINER,
            "--exec-argv-json",
            json.dumps(qualification_argv, separators=(",", ":")),
            "--expected-image-sha256",
            image_identity.sha256,
        )
    else:
        bootstrap_args = (
            "run",
            "--identity-record",
            _QUALIFICATION_CONTAINER,
            "--source-package",
            _SOURCE_PACKAGE_CONTAINER,
            "--toolkit-package",
            _TOOLKIT_PACKAGE_CONTAINER,
            "--exec-argv-json",
            json.dumps(qualification_argv, separators=(",", ":")),
            "--expected-image-sha256",
            image_identity.sha256,
        )
    governed_srun = render_governed_srun(
        image_identity,
        mounts=smoke_mounts,
        bootstrap_args=bootstrap_args,
    )
    compatibility_bootstrap_args = list(bootstrap_args)
    compatibility_bootstrap_args[compatibility_bootstrap_args.index("--exec-argv-json") + 1] = json.dumps(
        compatibility_argv, separators=(",", ":")
    )
    governed_compatibility_srun = render_governed_srun(
        image_identity,
        mounts=smoke_mounts,
        bootstrap_args=tuple(compatibility_bootstrap_args),
    )
    lines = [
        "#!/usr/bin/env bash",
        "# Runtime Qualification smoke job",
        f"# Tuple ID: {tuple_id}",
        f"#SBATCH --job-name=bspp_rq_{profile.name}_{attempt_token or tuple_id[:12]}",
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
    if resource.gres:
        lines.append(f"#SBATCH --gres={resource.gres}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            governed_compatibility_srun,
            governed_srun,
            "",
        ]
    )
    return "\n".join(lines)


def _toolkit_revision(qualification_tuple: Mapping[str, object]) -> str | None:
    selected = qualification_tuple.get("selected_source")
    if isinstance(selected, Mapping):
        revision = selected.get("revision")
        if isinstance(revision, str) and revision:
            return revision
    toolkit = qualification_tuple.get("toolkit_package_identity")
    revision = toolkit.get("commit") if isinstance(toolkit, Mapping) else None
    return revision if isinstance(revision, str) else None


def _valid_selected_source_identity(result_value: object, tuple_value: object) -> bool:
    if not isinstance(result_value, Mapping) or not isinstance(tuple_value, Mapping):
        return False
    if result_value.get("source_kind") != tuple_value.get("source_kind"):
        return False
    if result_value.get("root") != tuple_value.get("root"):
        return False
    if tuple_value.get("source_kind") == "baked":
        # Baked mode: the control plane cannot know the image's baked commit a
        # priori (only its sha256). Accept the smoke's self-reported revision,
        # structurally validated as a 40-hex sha; the image sha256 is the binding.
        revision = result_value.get("revision")
        return isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    return result_value.get("revision") == tuple_value.get("revision")


def _valid_runtime_ipsae_evidence(value: object, *, expected_revision: str | None, source_kind: str | None) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        evidence = runtime_ipsae_evidence_from_mapping(value)
    except ValueError:
        return False
    if source_kind == "baked":
        # Baked mode: the image's baked commit is authoritative (bound by the
        # image sha256); accept the smoke's self-reported revision, structurally
        # validated as a 40-hex sha.
        return (
            isinstance(evidence.source_revision, str)
            and len(evidence.source_revision) == 40
            and all(c in "0123456789abcdef" for c in evidence.source_revision)
        )
    return expected_revision is not None and evidence.source_revision == expected_revision


def _valid_publication_compatibility_evidence(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "check",
        "status",
        "process_count",
        "published_directory_count",
        "fallback_errno",
        "output_identity",
        "artifact",
    }:
        return False
    output_identity = value.get("output_identity")
    artifact = value.get("artifact")
    return (
        value.get("schema_version") == 1
        and value.get("check") == "postprocessing-directory-publication-v1"
        and value.get("status") == "passed"
        and value.get("process_count") == 2
        and value.get("published_directory_count") == 1
        and value.get("fallback_errno") == errno.EINVAL
        and isinstance(output_identity, Mapping)
        and set(output_identity) == {"path", "sha256", "size_bytes"}
        and output_identity.get("path") == "published/payload.json"
        and isinstance(output_identity.get("sha256"), str)
        and _SHA256.fullmatch(str(output_identity.get("sha256"))) is not None
        and type(output_identity.get("size_bytes")) is int
        and isinstance(artifact, Mapping)
        and set(artifact) == {"path", "sha256", "size_bytes"}
        and artifact.get("path") == "publication-compatibility.json"
        and isinstance(artifact.get("sha256"), str)
        and _SHA256.fullmatch(str(artifact.get("sha256"))) is not None
        and type(artifact.get("size_bytes")) is int
        and 0 < int(artifact.get("size_bytes", 0)) <= _MAX_PUBLICATION_COMPATIBILITY_BYTES
    )


def _publication_compatibility_artifact_matches(
    value: object,
    *,
    result_path: Path,
    remote_result_path: str | None,
    profile: ResolvedClusterProfile,
    runner: CommandRunner,
) -> bool:
    if not _valid_publication_compatibility_evidence(value):
        return False
    assert isinstance(value, Mapping)
    artifact = value["artifact"]
    assert isinstance(artifact, Mapping)
    local_path = result_path.parent / "publication-compatibility.json"
    observed: object = None
    if local_path.exists() or local_path.is_symlink():
        try:
            observed = _stable_artifact_identity(local_path, limit=_MAX_PUBLICATION_COMPATIBILITY_BYTES)
        except (OSError, ValueError):
            return False
    elif profile.transport == "ssh" and remote_result_path is not None:
        remote_path = str(Path(remote_result_path).parent / "publication-compatibility.json")
        probe = RemoteSlurmTransport(kind=profile.transport, ssh_target=profile.ssh_target, runner=runner).command(
            ("/usr/bin/python3", "-I", "-S", "-c", _REMOTE_RESULT_PROBE, remote_path, "4096")
        )
        try:
            observed = json.loads(probe.stdout) if probe.returncode == 0 else None
        except json.JSONDecodeError:
            return False
    return observed == {"size_bytes": artifact.get("size_bytes"), "sha256": artifact.get("sha256")}


def _publication_compatibility_launcher() -> str:
    return (
        "import importlib,sys;from pathlib import Path;"
        "source=Path(sys.argv[1]);module=sys.argv[2];"
        "sys.path[:0]=[str(source/'packages/orchestration-contract/src'),"
        "str(source/'packages/orchestration-runtime/src')];"
        "loaded=importlib.import_module(module);raise SystemExit(loaded.main(tuple(sys.argv[3:])))"
    )


def _runtime_ipsae_artifacts_match(
    value: object,
    *,
    result_path: Path,
    remote_result_path: str | None,
    profile: ResolvedClusterProfile,
    runner: CommandRunner,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        evidence = runtime_ipsae_evidence_from_mapping(value)
    except ValueError:
        return False
    transport = RemoteSlurmTransport(kind=profile.transport, ssh_target=profile.ssh_target, runner=runner)
    for artifact, limit in ((evidence.build_log, 1024 * 1024), (evidence.binary, 1024 * 1024 * 1024)):
        local_path = result_path.parent / artifact.path
        observed: object = None
        if local_path.exists() or local_path.is_symlink():
            try:
                observed = _stable_artifact_identity(local_path, limit=limit)
            except (OSError, ValueError):
                return False
        elif profile.transport == "ssh" and remote_result_path is not None:
            remote_path = str(Path(remote_result_path).parent / artifact.path)
            probe = transport.command(
                ("/usr/bin/python3", "-I", "-S", "-c", _REMOTE_RESULT_PROBE, remote_path, str(limit))
            )
            try:
                observed = json.loads(probe.stdout) if probe.returncode == 0 else None
            except json.JSONDecodeError:
                return False
        if observed != {"size_bytes": artifact.size_bytes, "sha256": artifact.sha256}:
            return False
    return True


def _stable_artifact_identity(path: Path, *, limit: int) -> dict[str, object]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("runtime iPSAE artifact is unsafe or oversize")
        digest = sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)

        def signature(item: os.stat_result) -> tuple[int, int, int, int, int]:
            return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)

        if signature(before) != signature(after) or signature(after) != signature(current):
            raise ValueError("runtime iPSAE artifact changed while reading")
        return {"size_bytes": size, "sha256": digest.hexdigest()}
    finally:
        os.close(descriptor)


def _runtime_ipsae_smoke_python() -> str:
    """Dependency-free qualification program run by isolated image Python."""
    return r"""import csv,hashlib,json,os,shutil,stat,subprocess,sys,tempfile
from pathlib import Path
LIMIT=65536
def run(argv):
 try:
  p=subprocess.run(tuple(argv),capture_output=True,text=True,check=False,env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':os.environ['HOME'],'LC_ALL':'C.UTF-8'})
  out=p.stdout.encode(errors='replace'); err=p.stderr.encode(errors='replace')
  return {'argv':list(argv),'returncode':p.returncode,
          'stdout':out[:LIMIT].decode(errors='replace'),'stderr':err[:LIMIT].decode(errors='replace'),
          'stdout_truncated':len(out)>LIMIT,'stderr_truncated':len(err)>LIMIT}
 except FileNotFoundError as e:
  return {'argv':list(argv),'returncode':127,'stdout':'','stderr':str(e)[:LIMIT],
          'stdout_truncated':False,'stderr_truncated':False}
def ok(e,label):
 if e['returncode']!=0: raise SystemExit(label+' failed: '+e['stderr'])
def identity(path,relative):
 b=path.lstat()
 if stat.S_ISLNK(b.st_mode) or not stat.S_ISREG(b.st_mode): raise SystemExit('unsafe runtime iPSAE artifact: '+relative)
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd); h=hashlib.sha256(); size=0
  while chunk:=os.read(fd,1048576): h.update(chunk); size+=len(chunk)
  after=os.fstat(fd)
 finally: os.close(fd)
 sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if sig(b)!=sig(opened) or sig(opened)!=sig(after): raise SystemExit('runtime iPSAE artifact changed while hashing')
 return {'path':relative,'sha256':h.hexdigest(),'size_bytes':size}
def bounded(path,relative,limit):
 b=path.lstat()
 if stat.S_ISLNK(b.st_mode) or not stat.S_ISREG(b.st_mode) or b.st_size>limit:
  raise SystemExit('unsafe or oversize bounded qualification artifact: '+relative)
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  opened=os.fstat(fd); chunks=[]; size=0; h=hashlib.sha256()
  while chunk:=os.read(fd,min(65536,limit+1-size)):
   chunks.append(chunk); size+=len(chunk); h.update(chunk)
   if size>limit: raise SystemExit('oversize bounded qualification artifact: '+relative)
  after=os.fstat(fd)
 finally: os.close(fd)
 current=path.lstat(); sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 if sig(b)!=sig(opened) or sig(opened)!=sig(after) or sig(after)!=sig(current):
  raise SystemExit('bounded qualification artifact changed while reading')
 return b''.join(chunks),{'path':relative,'sha256':h.hexdigest(),'size_bytes':size}
def sha(path):
 return identity(path,str(path))['sha256']
target=Path(sys.argv[1]); tuple_id=sys.argv[2]; bound=json.loads(sys.argv[3]); token=sys.argv[4]
toolkit=Path(sys.argv[5]); source_kind=sys.argv[6]; expected_revision=sys.argv[7]
relative=Path('afdb_integration_kit/ipsae'); source=toolkit/relative
makefile=source/'Makefile'
if not makefile.is_file() or makefile.is_symlink(): raise SystemExit('runtime iPSAE source lacks a safe Makefile')
if source_kind=='baked':
 provenance_path=toolkit/'provenance.json'
 if provenance_path.is_symlink() or not provenance_path.is_file():
  raise SystemExit('baked toolkit provenance attestation failed: missing or unsafe provenance file')
 try: provenance=json.loads(provenance_path.read_text())
 except (ValueError,OSError): raise SystemExit('baked toolkit provenance attestation failed: invalid JSON')
 commit=provenance.get('commit') if isinstance(provenance,dict) else None
 if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
  raise SystemExit('baked toolkit provenance attestation failed: commit is not a 40-hex sha')
 env_commit=os.environ.get('BSPP_EXPECTED_TOOLKIT_COMMIT')
 if env_commit is not None and commit!=env_commit:
  raise SystemExit('baked toolkit provenance attestation failed: commit does not match BSPP_EXPECTED_TOOLKIT_COMMIT')
 if not toolkit.is_dir() or toolkit.is_symlink():
  raise SystemExit('baked toolkit provenance attestation failed: root is not a safe directory')
 revision=commit
else:
 revision=bound['toolkit_package_identity']['commit']
retained=target.parent/'runtime-ipsae'; retained.mkdir(mode=0o700,parents=True,exist_ok=True)
built=source/'ipsae_cpp'
if built.exists() or built.is_symlink():
 mode=built.lstat().st_mode
 if stat.S_ISLNK(mode) or not stat.S_ISREG(mode): raise SystemExit('unsafe pre-existing private iPSAE target')
 built.unlink()
build_command=['make','-B','-C',str(source),'CXX=g++']
toolchain={'make':run(['make','--version']),'cxx':run(['g++','--version'])}
for name,evidence in toolchain.items(): ok(evidence,name+' version check')
build=run(build_command)
log=(('stdout:\n'+build['stdout']+'\nstderr:\n'+build['stderr']+'\n').encode(errors='replace')[:1048576])
log_path=retained/'build.log'
with log_path.open('xb') as f: f.write(log); f.flush(); os.fsync(f.fileno())
ok(build,'runtime iPSAE build')
binary=retained/'ipsae_cpp'
if not built.exists() or built.is_symlink() or not os.access(built,os.X_OK):
 raise SystemExit('runtime iPSAE build did not produce a safe executable')
with built.open('rb') as src,binary.open('xb') as dst: shutil.copyfileobj(src,dst); dst.flush(); os.fsync(dst.fileno())
binary.chmod(0o500); binary_evidence=identity(binary,'runtime-ipsae/ipsae_cpp')
version_output='ipsae_cpp source='+revision+' sha256='+binary_evidence['sha256']
version={'scheme':'source-revision+binary-sha256-v1','source_revision':revision,
         'binary_sha256':binary_evidence['sha256'],'output':version_output}
model_id=sys.argv[9]; pdb_text=sys.argv[10]; pae_text=sys.argv[11]; metadata_suffix=sys.argv[12]
expected_ab=sys.argv[13]; expected_ba=sys.argv[14]
fixture=retained/'functional-fixture'; fixture.mkdir(mode=0o700)
(fixture/(model_id+'-model_v1.pdb')).write_text(pdb_text)
(fixture/(model_id+metadata_suffix)).write_text(pae_text)
summary=retained/'functional-summary.csv'
functional=run([str(binary),'--batch',str(fixture),'10.0','8.0','--summary',str(summary),'--workers','1','--quiet'])
ok(functional,'runtime iPSAE functional test')
if not summary.is_file() or summary.is_symlink() or summary.stat().st_size == 0:
 raise SystemExit('runtime iPSAE functional test did not create its summary')
with summary.open(newline='') as handle: rows=list(csv.DictReader(handle))
if len(rows)!=1: raise SystemExit('runtime iPSAE functional test must produce exactly one result row')
row=rows[0]; observed_id=row.get('model_id') or Path(row.get('pdb_path','')).name.split('-model_',1)[0]
try:
 observed_ab=f"{float(row['ipsae_AB']):.6f}"; observed_ba=f"{float(row['ipsae_BA']):.6f}"
except (KeyError,TypeError,ValueError): raise SystemExit('runtime iPSAE functional result lacks directional scores')
if observed_id!=model_id or observed_ab!=expected_ab or observed_ba!=expected_ba:
 raise SystemExit('runtime iPSAE functional semantic result mismatch')
fixture_sha=hashlib.sha256(pdb_text.encode()+b'\0'+pae_text.encode()).hexdigest()
semantic=(json.dumps({'ipsae_AB':observed_ab,'ipsae_BA':observed_ba,'model_id':observed_id},sort_keys=True,separators=(',',':'))+'\n').encode()
functional.update({'fixture_sha256':fixture_sha,'result_sha256':hashlib.sha256(semantic).hexdigest(),
                   'check':'paired-pdb-pae-semantic-v1',
                   'expected_output':'one-row:'+model_id+':ipsae_AB='+expected_ab+':ipsae_BA='+expected_ba,
                   'status':'passed','model_id':observed_id,'ipsae_ab':observed_ab,'ipsae_ba':observed_ba})
gpu=run([sys.argv[8]]); ok(gpu,'GPU check')
runtime_ipsae={'format_version':1,'source_revision':revision,'source_path':relative.as_posix(),'build_command':build_command,'toolchain':toolchain,'build_result':build,'build_log':identity(log_path,'runtime-ipsae/build.log'),'binary':binary_evidence,'version':version,'functional_test':functional}
selected_source_identity={'source_kind':source_kind,'root':str(toolkit),'revision':revision}
if source_kind=='baked':
 selected_source_identity['image_identity']=bound['image_identity']
else:
 selected_source_identity['toolkit_package_identity']=bound['toolkit_package_identity']
bootstrap_sha256=sha(Path(sys.argv[15]))
compatibility_path=Path(sys.argv[16])
compatibility_bytes,compatibility_artifact=bounded(compatibility_path,'publication-compatibility.json',4096)
try: compatibility=json.loads(compatibility_bytes)
except (UnicodeDecodeError,ValueError): raise SystemExit('publication compatibility evidence is malformed')
expected_fields={'schema_version','check','status','process_count','published_directory_count','fallback_errno','output_identity'}
output_identity=compatibility.get('output_identity') if isinstance(compatibility,dict) else None
if (not isinstance(compatibility,dict) or set(compatibility)!=expected_fields or
 compatibility.get('schema_version')!=1 or compatibility.get('check')!='postprocessing-directory-publication-v1' or
 compatibility.get('status')!='passed' or compatibility.get('process_count')!=2 or
 compatibility.get('published_directory_count')!=1 or compatibility.get('fallback_errno')!=22 or
 not isinstance(output_identity,dict) or set(output_identity)!={'path','sha256','size_bytes'} or
 output_identity.get('path')!='published/payload.json'):
 raise SystemExit('publication compatibility evidence is invalid')
canonical=(json.dumps(compatibility,indent=2,sort_keys=True)+'\n').encode()
if compatibility_bytes!=canonical: raise SystemExit('publication compatibility evidence is not canonical')
compatibility=dict(compatibility)
compatibility['artifact']=compatibility_artifact
payload={'tuple_id':tuple_id,'job_id':os.environ.get('SLURM_JOB_ID'),'status':'succeeded','attempt_token':token,'python':sys.version,'gpu':gpu['stdout'],'bootstrap_sha256':bootstrap_sha256,'source_package_identity':bound['source_package_identity'],'toolkit_package_identity':bound.get('toolkit_package_identity'),'image_identity':bound['image_identity'],'selected_source_identity':selected_source_identity,'runtime_ipsae':runtime_ipsae,'publication_compatibility':compatibility}
fd,tmp=tempfile.mkstemp(dir=target.parent,prefix='.smoke.')
try: os.write(fd,(json.dumps(payload,sort_keys=True)+'\n').encode()); os.fsync(fd)
finally: os.close(fd)
os.replace(tmp,target); directory=os.open(target.parent,os.O_RDONLY)
try: os.fsync(directory)
finally: os.close(directory)
"""


def render_runtime_qualification_yaml(record: RuntimeQualificationRecord) -> str:
    """Render a compact qualification summary for CLI output."""
    payload = json.loads(record.path.read_text())
    smoke_job = payload.get("smoke_job")
    smoke_job_payload = smoke_job if isinstance(smoke_job, dict) else {}
    return yaml.safe_dump(
        {
            "runtime_qualification": {
                "profile": payload["profile"],
                "status": payload["status"],
                "tuple_id": payload["tuple_id"],
                "record_path": str(record.path),
                "expires_at": payload["expires_at"],
                "job_id": smoke_job_payload.get("job_id"),
                "script_path": smoke_job_payload.get("script_path"),
            }
        },
        sort_keys=False,
    )


def render_runtime_qualification_check_yaml(check: RuntimeQualificationCheck) -> str:
    """Render one check without reopening its mutable authority pathname."""
    summary: dict[str, object] = {
        "current": check.current,
        "reason": check.reason,
        "tuple_id": check.tuple_id,
        "record_path": str(check.record_path),
    }
    if check.current:
        if check.snapshot is None:
            raise ValueError("current Runtime Qualification check lacks an exact record snapshot")
        payload = json.loads(check.snapshot.document)
        if not isinstance(payload, Mapping):
            raise ValueError("Runtime Qualification snapshot must be a mapping")
        smoke_job = payload.get("smoke_job")
        smoke_job_mapping = smoke_job if isinstance(smoke_job, Mapping) else {}
        summary.update(
            {
                "profile": payload.get("profile"),
                "status": payload.get("status"),
                "evidence_status": payload.get("evidence_status"),
                "record_sha256": check.snapshot.sha256,
                "record_size_bytes": check.snapshot.size_bytes,
                "qualified_at": payload.get("qualified_at"),
                "expires_at": payload.get("expires_at"),
                "job_id": smoke_job_mapping.get("job_id"),
                "script_path": smoke_job_mapping.get("script_path"),
            }
        )
    return yaml.safe_dump({"runtime_qualification": summary}, sort_keys=False)


def _runtime_qualification_container_script(
    *,
    qualification_tuple: Mapping[str, object],
    tuple_id: str,
    record_path: Path,
    source_bundle_path: Path,
    image_path: str,
    expires_hours: int,
) -> str:
    tuple_json = json.dumps(dict(qualification_tuple), sort_keys=True, separators=(",", ":"))
    lines = [
        "set -euo pipefail",
        "",
        f"export BSPP_RUNTIME_QUALIFICATION_RECORD={shlex.quote(str(record_path))}",
        f"export BSPP_RUNTIME_QUALIFICATION_TUPLE_ID={shlex.quote(tuple_id)}",
        f"export BSPP_RUNTIME_QUALIFICATION_TUPLE_JSON={shlex.quote(tuple_json)}",
        f"export BSPP_RUNTIME_QUALIFICATION_EXPIRES_HOURS={expires_hours}",
        f"export BSPP_SOURCE_BUNDLE={shlex.quote(str(source_bundle_path))}",
        f"export BSPP_EXECUTION_RUNTIME_IMAGE={shlex.quote(image_path)}",
        f"export TOOLKIT_ROOT={shlex.quote(str(TOOLKIT_CONTAINER_ROOT))}",
        f'PYTHON_BIN="${{BSPP_ORCHESTRATION_PYTHON:-{PIXI_PYTHON_PATH}}}"',
        'fail() { echo "BSPP Runtime Qualification failed: $*" >&2; exit 127; }',
        'require_executable() { [[ -x "$1" ]] || fail "missing executable $2: $1"; }',
        'require_file() { [[ -f "$1" ]] || fail "missing file $2: $1"; }',
        'require_dir() { [[ -d "$1" ]] || fail "missing directory $2: $1"; }',
        "",
        'require_file "$BSPP_SOURCE_BUNDLE" "Source Bundle"',
        'require_executable "$PYTHON_BIN" "Python runtime"',
        'SOURCE_WORKDIR="${SLURM_TMPDIR:-/tmp}/bspp-runtime-qualification-source"',
        'rm -rf "$SOURCE_WORKDIR"',
        'mkdir -p "$SOURCE_WORKDIR"',
        'tar -xaf "$BSPP_SOURCE_BUNDLE" -C "$SOURCE_WORKDIR"',
        'require_dir "$SOURCE_WORKDIR/packages/orchestration-contract/src/bspp/orchestration/contract" '
        '"control contract package from Source Bundle"',
        'require_dir "$SOURCE_WORKDIR/packages/orchestration-runtime/src/bspp/orchestration/runtime" '
        '"runtime package from Source Bundle"',
        'require_dir "$TOOLKIT_ROOT" "AFDB Toolkit Source"',
        'export BSPP_EXTRACTED_SOURCE_ROOT="$SOURCE_WORKDIR"',
        "",
        "\"$PYTHON_BIN\" - <<'BSPP_WRITE_RUNTIME_QUALIFICATION'",
        _runtime_qualification_writer_python(),
        "BSPP_WRITE_RUNTIME_QUALIFICATION",
    ]
    return "\n".join(lines)


def _runtime_qualification_writer_python() -> str:
    return r"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def run(argv):
    try:
        result = subprocess.run(tuple(argv), capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return {
            "argv": list(argv),
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "argv": list(argv),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path):
    result = run(("git", "-C", str(path), "rev-parse", "HEAD"))
    return result["stdout"] if result["returncode"] == 0 and result["stdout"] else None


record_path = Path(os.environ["BSPP_RUNTIME_QUALIFICATION_RECORD"])
qualification_tuple = json.loads(os.environ["BSPP_RUNTIME_QUALIFICATION_TUPLE_JSON"])
tuple_id = os.environ["BSPP_RUNTIME_QUALIFICATION_TUPLE_ID"]
source_bundle = Path(os.environ["BSPP_SOURCE_BUNDLE"])
source_root = Path(os.environ["BSPP_EXTRACTED_SOURCE_ROOT"])
toolkit_root = Path(os.environ["TOOLKIT_ROOT"])
expires_hours = int(os.environ["BSPP_RUNTIME_QUALIFICATION_EXPIRES_HOURS"])
qualified_at = datetime.now(tz=UTC).replace(microsecond=0)
expires_at = qualified_at + timedelta(hours=expires_hours)
gpu = run(("nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"))
gpu_gres = qualification_tuple.get("runtime_facts", {}).get("gpu_worker_gres")
if isinstance(gpu_gres, str) and "gpu" in gpu_gres.lower() and gpu["returncode"] != 0:
    raise SystemExit("nvidia-smi did not return GPU evidence for Runtime Qualification")
payload = {
    "schema_version": 1,
    "profile": qualification_tuple["cluster_profile"],
    "status": "qualified",
    "evidence_status": "qualified",
    "qualified_at": qualified_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "tuple_id": tuple_id,
    "tuple": qualification_tuple,
    "smoke_evidence": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "execution_runtime_image": os.environ["BSPP_EXECUTION_RUNTIME_IMAGE"],
        },
        "gpu": gpu,
        "source": {
            "bundle_path": str(source_bundle),
            "bundle_sha256": sha256_file(source_bundle),
            "extracted_source_root": str(source_root),
            "control_contract_package_present": (
                source_root
                / "packages"
                / "orchestration-contract"
                / "src"
                / "bspp"
                / "orchestration"
                / "contract"
            ).is_dir(),
            "runtime_package_present": (
                source_root
                / "packages"
                / "orchestration-runtime"
                / "src"
                / "bspp"
                / "orchestration"
                / "runtime"
            ).is_dir(),
        },
        "toolkit": {
            "path": str(toolkit_root),
            "exists": toolkit_root.is_dir(),
            "git_head": git_head(toolkit_root),
        },
    },
}
tmp_path = record_path.with_name(f".{record_path.name}.tmp")
record_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp_path.replace(record_path)
""".strip()


def _record_path(root: str, *, profile_name: str, tuple_id: str) -> Path:
    return Path(root) / profile_name / f"{tuple_id}.json"


def _expected_attempt_paths(
    *,
    profile: ResolvedClusterProfile,
    tuple_id: str,
    attempt_token: str,
    control_root: str,
    remote_root: str,
) -> dict[str, Path]:
    if profile.runtime_qualification_control_root != control_root or profile.runtime_qualification_root != remote_root:
        raise ValueError("Runtime Qualification attempt roots differ from the selected profile")
    return expected_runtime_qualification_attempt_paths(
        profile=profile,
        tuple_id=tuple_id,
        attempt_token=attempt_token,
    )


def _validate_persisted_attempt_binding(
    payload: Mapping[str, object],
    *,
    profile: ResolvedClusterProfile,
    tuple_id: str,
    control_root: str,
    remote_root: str,
    record_path: Path,
) -> None:
    if profile.runtime_qualification_control_root != control_root or profile.runtime_qualification_root != remote_root:
        raise ValueError("Runtime Qualification attempt roots differ from the selected profile")
    validate_runtime_qualification_attempt_binding(
        payload,
        profile=profile,
        tuple_id=tuple_id,
        record_path=record_path,
    )


def _submitted_payload(
    *,
    profile: ResolvedClusterProfile,
    qualification_tuple: dict[str, object],
    tuple_id: str,
    record_path: Path,
    script_path: Path,
    submitted_at: datetime,
    expires_hours: int,
    job_id: str | None,
    submit_command: tuple[str, ...],
    attempt_token: str | None = None,
    input_path: Path | None = None,
    result_path: Path | None = None,
    remote_script_path: Path | None = None,
    remote_input_path: Path | None = None,
    remote_result_path: Path | None = None,
    status: str = "submitted",
    attempt_history: list[object] | None = None,
) -> dict[str, object]:
    expires_at = submitted_at + timedelta(hours=expires_hours)
    return {
        "schema_version": 1,
        "profile": profile.name,
        "status": status,
        "evidence_status": "pending",
        "submitted_at": _format_instant(submitted_at),
        "qualified_at": None,
        "expires_at": _format_instant(expires_at),
        "tuple_id": tuple_id,
        "tuple": qualification_tuple,
        "attempt_history": list(attempt_history or ()),
        "smoke_job": {
            "job_id": job_id,
            "script_path": str(script_path),
            "submit_command": list(submit_command),
            "record_path": str(record_path),
            "attempt_token": attempt_token,
            "input_path": str(input_path) if input_path is not None else str(record_path),
            "result_path": str(result_path) if result_path is not None else None,
            "remote_script_path": str(remote_script_path) if remote_script_path is not None else str(script_path),
            "remote_input_path": str(remote_input_path) if remote_input_path is not None else str(record_path),
            "remote_result_path": str(remote_result_path) if remote_result_path is not None else None,
        },
    }


def _qualified_payload(
    *,
    profile: ResolvedClusterProfile,
    qualification_tuple: dict[str, object],
    tuple_id: str,
    qualified_at: datetime,
    expires_hours: int,
    smoke_evidence: Mapping[str, object],
) -> dict[str, object]:
    expires_at = qualified_at + timedelta(hours=expires_hours)
    return {
        "schema_version": 1,
        "profile": profile.name,
        "status": "qualified",
        "evidence_status": "qualified",
        "qualified_at": _format_instant(qualified_at),
        "expires_at": _format_instant(expires_at),
        "tuple_id": tuple_id,
        "tuple": qualification_tuple,
        "smoke_evidence": dict(smoke_evidence),
    }


def _smoke_container_mounts(
    *,
    source_bundle_path: Path,
    toolkit_package_path: Path | None = None,
    record_path: Path,
    result_path: Path,
) -> str:
    mounts = [
        (str(source_bundle_path), f"{_SOURCE_PACKAGE_CONTAINER}:ro"),
        (str(record_path), f"{_QUALIFICATION_CONTAINER}:ro"),
        (str(result_path.parent), "/run/bspp/result"),
    ]
    if toolkit_package_path is not None:
        mounts.insert(1, (str(toolkit_package_path), f"{_TOOLKIT_PACKAGE_CONTAINER}:ro"))
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


def _tuple_path(qualification_tuple: Mapping[str, object], key: str) -> Path:
    return Path(_tuple_string(qualification_tuple, key))


def _tuple_string(qualification_tuple: Mapping[str, object], key: str) -> str:
    value = qualification_tuple.get(key)
    if not isinstance(value, str) or not value:
        msg = f"Runtime Qualification tuple is missing {key}"
        raise ValueError(msg)
    return value


def _selected_source_kind(profile: ResolvedClusterProfile) -> str:
    """Return 'baked' when toolkit repo is empty/falsy — baked image mode."""
    if not profile.afdb_toolkit_repo:
        return "baked"
    return "override"


def _qualification_tuple(
    profile: ResolvedClusterProfile,
    *,
    source_repo: Path,
    source_package_identity_path: Path | None = None,
    source_package_identity: SourcePackageIdentity | None = None,
    runner: CommandRunner = default_command_runner,
) -> dict[str, object]:
    if profile.transport == "ssh" and source_package_identity_path is None and source_package_identity is None:
        raise ValueError("SSH Runtime Qualification requires --source-package-identity")
    source_kind = _selected_source_kind(profile)
    source_identity = _source_identity(source_repo, profile=profile)
    supplied_source_package = source_package_identity is not None or source_package_identity_path is not None
    gpu_worker = profile.resources.get(_SCHEDULING_CLASS)
    cache_path = None
    if profile.runtime_image_policy == "trusted-cache":
        cache_root = _require_profile_value(profile.runtime_image_cache_root, "runtime_image_cache_root")
        cache_path = Path(cache_root) / Path(profile.image).name
    remote_identities: tuple[ImageIdentity, SourcePackageIdentity] | None = None
    if profile.transport == "ssh":
        if source_kind == "override":
            remote_identities = _remote_image_and_toolkit_identities(profile, runner=runner)
            image_identity = remote_identities[0]
        else:  # baked
            image_identity = _remote_image_identity(profile, runner=runner)
    else:
        image_identity = identify_runtime_image(
            Path(profile.image), policy=profile.runtime_image_policy, trusted_cache=cache_path
        )
    if source_package_identity is not None:
        source_package = source_package_identity
    elif source_package_identity_path is None:
        source_package = _ensure_source_package_identity(source_repo, source_identity)
    else:
        raw_identity = yaml.safe_load(source_package_identity_path.read_text())
        if not isinstance(raw_identity, Mapping):
            raise ValueError("Source package identity record must be a mapping")
        wrapped_identity = raw_identity.get("source_package_identity")
        if isinstance(wrapped_identity, Mapping):
            raw_identity = wrapped_identity
        source_package = source_package_identity_from_mapping(raw_identity)
    if supplied_source_package:
        _validate_source_package_against_checkout(
            profile,
            source_repo=source_repo,
            source_identity=source_identity,
            package_identity=source_package,
        )
    _verify_staged_source_package(profile, source_package, runner=runner)
    common: dict[str, object] = {
        "cluster_profile": profile.name,
        "scheduling_class": _SCHEDULING_CLASS,
        "execution_runtime_image": str(image_identity.path),
        "runtime_image_policy": image_identity.policy,
        "runtime_image_size_bytes": image_identity.size_bytes,
        "runtime_image_sha256": image_identity.sha256,
        "image_identity": image_identity.to_mapping(),
        "runtime_facts": {
            "gpu_worker_gres": gpu_worker.gres if gpu_worker is not None else None,
        },
        "source_bundle_id": source_identity["bundle_id"],
        "source_bundle_path": str(source_package.package_path),
        "source_package_identity": source_package.to_mapping(),
    }
    if source_kind == "baked":
        common["selected_source"] = {
            "source_kind": "baked",
            "root": str(BAKED_TOOLKIT_CONTAINER_ROOT),
            "revision": BAKED_TOOLKIT_COMMIT,
            "image_identity": image_identity.to_mapping(),
        }
    else:
        toolkit_package = remote_identities[1] if remote_identities is not None else _local_toolkit_package(profile)
        common["toolkit_source"] = profile.afdb_toolkit_repo
        common["toolkit_package_identity"] = toolkit_package.to_mapping()
        common["toolkit_package_path"] = str(toolkit_package.package_path)
        common["selected_source"] = {
            "source_kind": "override",
            "root": str(TOOLKIT_CONTAINER_ROOT),
            "revision": toolkit_package.commit,
            "toolkit_package_identity": toolkit_package.to_mapping(),
        }
    return common


def _discover_source_package_identity(
    profile: ResolvedClusterProfile, *, source_repo: Path
) -> SourcePackageIdentity | None:
    if profile.transport != "ssh" or profile.runtime_qualification_control_root is None:
        return None
    commit = _git(source_repo.resolve(), "rev-parse", "HEAD")
    records = sorted((Path(profile.runtime_qualification_control_root) / profile.name).glob("*.json"), reverse=True)
    for record in records:
        try:
            payload = json.loads(record.read_text())
            tuple_payload = payload.get("tuple") if isinstance(payload, dict) else None
            raw = tuple_payload.get("source_package_identity") if isinstance(tuple_payload, dict) else None
            if isinstance(raw, Mapping):
                identity = source_package_identity_from_mapping(raw)
                if identity.commit == commit:
                    return identity
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _source_identity(source_repo: Path, *, profile: ResolvedClusterProfile) -> dict[str, str]:
    source_repo = source_repo.resolve()
    commit = _git(source_repo, "rev-parse", "HEAD")
    tree = _git(source_repo, "rev-parse", f"{commit}^{{tree}}")
    status = _git(source_repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        msg = "Runtime Qualification requires a committed source repository"
        raise RuntimeQualificationSourceError("dirty-source", msg)
    bundle_id = f"bspp-orchestration-{commit}"
    source_bundle_root = _require_profile_value(profile.source_bundle_root, "source_bundle_root")
    return {
        "bundle_id": bundle_id,
        "target_path": str(Path(source_bundle_root) / f"{bundle_id}.tar"),
        "commit": commit,
        "tree": tree,
    }


def _validate_source_package_against_checkout(
    profile: ResolvedClusterProfile,
    *,
    source_repo: Path,
    source_identity: Mapping[str, str],
    package_identity: SourcePackageIdentity,
) -> None:
    """Bind a supplied package identity to the current clean governed source snapshot."""
    commit = source_identity["commit"]
    tree = source_identity["tree"]
    if package_identity.commit != commit:
        raise ValueError("Supplied source package commit does not match clean source checkout")
    if package_identity.tree != tree:
        raise ValueError("Supplied source package tree does not match clean source checkout")

    package_root = profile.governed_package_root
    if profile.transport == "ssh":
        package_root = _require_profile_value(package_root, "governed_package_root")
    if package_root is not None:
        expected_path = Path(package_root) / "orchestration" / commit / f"{package_identity.package_sha256}.tar"
        if package_identity.package_path != expected_path:
            raise ValueError("Supplied source package is not at the current checkout digest-addressed path")

    with tempfile.TemporaryDirectory(prefix="bspp-runtime-qualification-") as temporary:
        expected = build_source_package(
            source_repo,
            Path(temporary) / "source-package.tar",
            commit=commit,
            tree=tree,
            tracked_git=True,
        )
    if package_identity.manifest_sha256 != expected.manifest_sha256:
        raise ValueError("Supplied source package manifest does not match clean source checkout")


def _local_toolkit_package(profile: ResolvedClusterProfile) -> SourcePackageIdentity:
    toolkit_repo_str = _require_profile_value(profile.afdb_toolkit_repo, "afdb_toolkit_repo")
    toolkit_repo = Path(toolkit_repo_str).resolve(strict=True)
    commit = _git(toolkit_repo, "rev-parse", "HEAD")
    tree = _git(toolkit_repo, "rev-parse", "HEAD^{tree}")
    package_root = Path(_require_profile_value(profile.governed_package_root, "governed_package_root"))
    build_root = package_root / ".build"
    build_root.mkdir(parents=True, exist_ok=True)
    candidate = build_root / f"toolkit-{commit}-{uuid.uuid4().hex}.tar"
    package = build_source_package(
        toolkit_repo,
        candidate,
        commit=commit,
        tree=tree,
        tracked_git=True,
        governed_runtime_only=False,
        allow_untracked=True,
        package_role="toolkit",
    )
    target = package_root / "toolkit" / commit / f"{package.package_sha256}.tar"
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate_fd = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(candidate_fd)
    finally:
        os.close(candidate_fd)
    try:
        os.link(candidate, target)
    except FileExistsError:
        existing = SourcePackageIdentity(**{**package.__dict__, "package_path": target.absolute()})
        verify_source_package(existing)
    finally:
        candidate.unlink(missing_ok=True)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return SourcePackageIdentity(**{**package.__dict__, "package_path": target.absolute()})


def _remote_image_and_toolkit_identities(
    profile: ResolvedClusterProfile, *, runner: CommandRunner
) -> tuple[ImageIdentity, SourcePackageIdentity]:
    package_root = _require_profile_value(profile.governed_package_root, "governed_package_root")
    toolkit_repo_str = _require_profile_value(profile.afdb_toolkit_repo, "afdb_toolkit_repo")
    command = command_argv(
        (
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            _REMOTE_IDENTITY_PRODUCER,
            toolkit_repo_str,
            profile.image,
            package_root,
            profile.runtime_image_policy,
            str(
                Path(profile.runtime_image_cache_root) / Path(profile.image).name
                if profile.runtime_image_policy == "trusted-cache" and profile.runtime_image_cache_root
                else ""
            ),
        ),
        transport=profile.transport,
        ssh_target=profile.ssh_target,
    )
    result = runner(command)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "remote governed identity producer failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("remote governed identity producer returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("remote governed identity producer returned invalid identity payload")
    image_value = payload.get("image_identity")
    toolkit_value = payload.get("toolkit_package_identity")
    if not isinstance(image_value, Mapping) or not isinstance(toolkit_value, Mapping):
        raise ValueError("remote governed identity producer omitted image or toolkit package identity")
    image_identity = image_identity_from_mapping(image_value)
    if image_identity.policy != profile.runtime_image_policy:
        raise ValueError("remote image identity policy mismatch")
    return image_identity, source_package_identity_from_mapping(toolkit_value)


def _remote_image_identity(profile: ResolvedClusterProfile, *, runner: CommandRunner) -> ImageIdentity:
    """Identify the Execution Runtime image on the remote cluster for baked mode.

    Baked mode has no toolkit repo, so the override-mode producer (which packages
    the toolkit) cannot be used. This runs a remote image-only identity producer
    over the configured SSH transport.
    """
    cache = str(
        Path(profile.runtime_image_cache_root) / Path(profile.image).name
        if profile.runtime_image_policy == "trusted-cache" and profile.runtime_image_cache_root
        else ""
    )
    command = command_argv(
        (
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            _REMOTE_IMAGE_IDENTITY_PRODUCER,
            profile.image,
            profile.runtime_image_policy,
            cache,
        ),
        transport=profile.transport,
        ssh_target=profile.ssh_target,
    )
    result = runner(command)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "remote image identity producer failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("remote image identity producer returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("remote image identity producer returned invalid payload")
    image_value = payload.get("image_identity")
    if not isinstance(image_value, Mapping):
        raise ValueError("remote image identity producer omitted image identity")
    image_identity = image_identity_from_mapping(image_value)
    if image_identity.policy != profile.runtime_image_policy:
        raise ValueError("remote image identity policy mismatch")
    return image_identity


_REMOTE_IMAGE_IDENTITY_PRODUCER = r"""import hashlib,json,os,stat,sys
image,policy,cache=sys.argv[1:]
def bind_image(path):
 before=os.lstat(path)
 if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): raise SystemExit('image unsafe')
 canonical=os.path.realpath(path); fd=os.open(canonical,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  st=os.fstat(fd); h=hashlib.sha256()
  if not stat.S_ISREG(st.st_mode): raise SystemExit('image nonregular')
  while True:
   chunk=os.read(fd,1048576)
   if not chunk: break
   h.update(chunk)
  after_fd=os.fstat(fd)
 finally: os.close(fd)
 after=os.lstat(path); canonical_stat=os.lstat(canonical)
 signatures=((before.st_dev,before.st_ino),(after.st_dev,after.st_ino),(st.st_dev,st.st_ino),
  (after_fd.st_dev,after_fd.st_ino),(canonical_stat.st_dev,canonical_stat.st_ino))
 changed=(st.st_size,st.st_mtime_ns)!=(after_fd.st_size,after_fd.st_mtime_ns)
 if len(set(signatures))!=1 or changed: raise SystemExit('image changed')
 return canonical,st.st_size,h.hexdigest()
original=bind_image(image)
if policy=='trusted-cache':
 if not cache: raise SystemExit('trusted-cache path required')
 selected=bind_image(cache)
 if original[1:]!=selected[1:]: raise SystemExit('trusted-cache identity mismatch')
elif policy=='digest-checked': selected=original
else: raise SystemExit('unknown image policy')
image_identity={'format_version':1,'policy':policy,'path':selected[0],
 'size_bytes':selected[1],'sha256':selected[2]}
print(json.dumps({'image_identity':image_identity},sort_keys=True))
"""


_REMOTE_IDENTITY_PRODUCER = r"""import hashlib,json,os,stat,subprocess,sys,tarfile,tempfile
repo,image,root,policy,cache=sys.argv[1:]
def git(*a): return subprocess.run(('git','-C',repo,*a),check=True,capture_output=True,text=True).stdout.strip()
if git('status','--porcelain=v1','--untracked-files=no'): raise SystemExit('dirty tracked toolkit')
commit=git('rev-parse','HEAD'); tree=git('rev-parse','HEAD^{tree}')
raw=subprocess.run(('git','-C',repo,'ls-files','-z','--stage'),check=True,capture_output=True).stdout
entries=[]
for record in raw.split(b'\0'):
 if not record: continue
 meta,name=record.split(b'\t',1); mode,oid,stage=meta.decode().split(); path=name.decode('utf-8')
 if stage!='0' or mode not in ('100644','100755'): raise SystemExit('nonregular toolkit member')
 full=os.path.join(repo,*path.split('/')); fd=os.open(full,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd); h=hashlib.sha256(); size=0
  while True:
   chunk=os.read(fd,1048576)
   if not chunk: break
   size+=len(chunk); h.update(chunk)
  after=os.fstat(fd)
  before_sig=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
  after_sig=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  if before_sig!=after_sig: raise SystemExit('mutated toolkit')
 finally: os.close(fd)
 entries.append({'mode':'0o755' if mode=='100755' else '0o644','path':path,'sha256':h.hexdigest(),'size_bytes':size})
entries.sort(key=lambda e:e['path'].encode())
manifest=(json.dumps({'commit':commit,'entries':entries,'tree':tree},sort_keys=True,separators=(',',':'))+'\n').encode()
build=os.path.join(root,'.build'); os.makedirs(build,mode=0o700,exist_ok=True)
fd,tmp=tempfile.mkstemp(prefix='toolkit-',suffix='.tar',dir=build); os.close(fd); os.unlink(tmp)
with tarfile.open(tmp,'x',format=tarfile.USTAR_FORMAT) as tar:
 for name,data,mode in [('.bspp-source-manifest.json',manifest,0o600)]:
  info=tarfile.TarInfo(name); info.size=len(data); info.mode=mode
  info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
  import io; tar.addfile(info,io.BytesIO(data))
 for entry in entries:
  info=tarfile.TarInfo(entry['path']); info.size=entry['size_bytes']; info.mode=int(entry['mode'],8)
  info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
  member=os.path.join(repo,*entry['path'].split('/')); member_fd=os.open(member,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   member_before=os.fstat(member_fd)
   with os.fdopen(os.dup(member_fd),'rb') as stream: tar.addfile(info,stream)
   member_after=os.fstat(member_fd)
   member_before_sig=(member_before.st_dev,member_before.st_ino,member_before.st_size,member_before.st_mtime_ns)
   member_after_sig=(member_after.st_dev,member_after.st_ino,member_after.st_size,member_after.st_mtime_ns)
   if member_before_sig!=member_after_sig: raise SystemExit('mutated toolkit')
  finally: os.close(member_fd)
package_hash=hashlib.sha256()
with open(tmp,'rb') as stream:
 while True:
  chunk=stream.read(1048576)
  if not chunk: break
  package_hash.update(chunk)
package_sha=package_hash.hexdigest()
size=os.stat(tmp).st_size
tmp_fd=os.open(tmp,os.O_RDONLY|os.O_NOFOLLOW); os.fsync(tmp_fd); os.close(tmp_fd)
final=os.path.join(root,'toolkit',commit,package_sha+'.tar')
os.makedirs(os.path.dirname(final),exist_ok=True)
try:
 try: os.link(tmp,final)
 except FileExistsError:
  fd=os.open(final,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   final_stat=os.fstat(fd); final_hash=hashlib.sha256()
   while True:
    chunk=os.read(fd,1048576)
    if not chunk: break
    final_hash.update(chunk)
  finally: os.close(fd)
  if not stat.S_ISREG(final_stat.st_mode) or (final_stat.st_size,final_hash.hexdigest())!=(size,package_sha):
   raise SystemExit('existing toolkit package diverges')
 directory=os.open(os.path.dirname(final),os.O_RDONLY)
 try: os.fsync(directory)
 except OSError: pass
 finally: os.close(directory)
finally: os.unlink(tmp)
def bind_image(path):
 before=os.lstat(path)
 if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): raise SystemExit('image unsafe')
 canonical=os.path.realpath(path); fd=os.open(canonical,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  st=os.fstat(fd); h=hashlib.sha256()
  if not stat.S_ISREG(st.st_mode): raise SystemExit('image nonregular')
  while True:
   chunk=os.read(fd,1048576)
   if not chunk: break
   h.update(chunk)
  after_fd=os.fstat(fd)
 finally: os.close(fd)
 after=os.lstat(path); canonical_stat=os.lstat(canonical)
 signatures=((before.st_dev,before.st_ino),(after.st_dev,after.st_ino),(st.st_dev,st.st_ino),
  (after_fd.st_dev,after_fd.st_ino),(canonical_stat.st_dev,canonical_stat.st_ino))
 changed=(st.st_size,st.st_mtime_ns)!=(after_fd.st_size,after_fd.st_mtime_ns)
 if len(set(signatures))!=1 or changed: raise SystemExit('image changed')
 return canonical,st.st_size,h.hexdigest()
original=bind_image(image)
if policy=='trusted-cache':
 if not cache: raise SystemExit('trusted-cache path required')
 selected=bind_image(cache)
 if original[1:]!=selected[1:]: raise SystemExit('trusted-cache identity mismatch')
elif policy=='digest-checked': selected=original
else: raise SystemExit('unknown image policy')
image_identity={'format_version':1,'policy':policy,'path':selected[0],
 'size_bytes':selected[1],'sha256':selected[2]}
toolkit_identity={'format_version':1,'format':'bspp-tar-v1','verifier':'safe-tar-v1',
 'package_path':final,'package_size_bytes':size,'package_sha256':package_sha,
 'manifest_sha256':hashlib.sha256(manifest).hexdigest(),'commit':commit,'tree':tree,
 'package_role':'toolkit','policy_version':1}
print(json.dumps({'image_identity':image_identity,'toolkit_package_identity':toolkit_identity},sort_keys=True))
"""


def _tuple_int(qualification_tuple: Mapping[str, object], key: str) -> int:
    value = qualification_tuple.get(key)
    if type(value) is not int:
        msg = f"Runtime Qualification tuple is missing {key}"
        raise ValueError(msg)
    return value


def _ensure_source_package_identity(source_repo: Path, source_identity: Mapping[str, str]) -> SourcePackageIdentity:
    target = Path(source_identity["target_path"]).absolute()
    commit = source_identity["commit"]
    tree = source_identity["tree"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        return build_source_package(source_repo, target, commit=commit, tree=tree, tracked_git=True)
    with tarfile.open(target, "r:") as archive:
        member = archive.getmember(MANIFEST_MEMBER)
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Existing source package manifest is unreadable")
        manifest_sha256 = sha256(stream.read()).hexdigest()
    existing = SourcePackageIdentity.for_existing_archive(
        target,
        format="bspp-tar-v1",
        verifier="safe-tar-v1",
        manifest_sha256=manifest_sha256,
        commit=commit,
        tree=tree,
    )
    return SourcePackageIdentity(**{**existing.__dict__, "package_role": "orchestration"})


def _verify_staged_source_package(
    profile: ResolvedClusterProfile, identity: SourcePackageIdentity, *, runner: CommandRunner
) -> None:
    if identity.package_role != "orchestration" or identity.policy_version != 1:
        raise ValueError("staged source package has invalid orchestration role policy")
    if profile.transport != "ssh":
        verify_source_package(identity, expected_role="orchestration")
        return
    policy_command = command_argv(
        (
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            _REMOTE_SOURCE_POLICY_VERIFIER,
            str(identity.package_path),
            str(identity.package_size_bytes),
            identity.package_sha256,
            identity.manifest_sha256,
        ),
        transport=profile.transport,
        ssh_target=profile.ssh_target,
    )
    policy_result = runner(policy_command)
    if policy_result.returncode != 0 or policy_result.stdout.strip() != "OK":
        raise ValueError("staged source package violates orchestration payload policy")


_REMOTE_SOURCE_POLICY_VERIFIER = r"""import hashlib,json,os,stat,sys,tarfile
path,expected_size,expected_sha,expected_manifest=sys.argv[1:]; expected_size=int(expected_size)
allowed_prefixes=('packages/orchestration-contract/src/','packages/orchestration-runtime/src/')
allowed_files={'containers/scripts/slurm-semantic-acceptance.sh','containers/scripts/slurm-tar-payload-parity.sh'}
if not os.path.isabs(path) or os.path.normpath(path)!=path or os.path.realpath(path)!=path:
 raise SystemExit('package path uses symlink or lexical alias')
before=os.lstat(path)
if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode): raise SystemExit('unsafe package')
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
try:
 opened=os.fstat(fd); digest=hashlib.sha256()
 while chunk:=os.read(fd,1048576): digest.update(chunk)
 if opened.st_size!=expected_size or digest.hexdigest()!=expected_sha: raise SystemExit('package identity mismatch')
 os.lseek(fd,0,os.SEEK_SET)
 archive=tarfile.open(fileobj=os.fdopen(os.dup(fd),'rb'),mode='r:')
 member=archive.next()
 if member is None or member.name!='.bspp-source-manifest.json' or member.size>67108864:
  raise SystemExit('source package policy manifest invalid')
 stream=archive.extractfile(member)
 if stream is None: raise SystemExit('source package policy manifest missing')
 manifest=stream.read(member.size+1)
 archive.close(); after=os.fstat(fd)
finally: os.close(fd)
current=os.lstat(path); sig=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
if sig(before)!=sig(opened) or sig(opened)!=sig(after) or sig(after)!=sig(current): raise SystemExit('changed package')
if len(manifest)!=member.size or hashlib.sha256(manifest).hexdigest()!=expected_manifest:
 raise SystemExit('source package policy manifest mismatch')
payload=json.loads(manifest)
for entry in payload['entries']:
 path=entry['path']
 if not path.startswith(allowed_prefixes) and path not in allowed_files:
  raise SystemExit('source package policy member rejected')
print('OK')
"""


def _tuple_id(qualification_tuple: dict[str, object]) -> str:
    canonical = json.dumps(qualification_tuple, sort_keys=True, separators=(",", ":")).encode()
    return sha256(canonical).hexdigest()


def _coerce_utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _format_instant(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_instant(value: object) -> datetime:
    if not isinstance(value, str):
        msg = "Runtime Qualification record is missing expires_at"
        raise ValueError(msg)
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        msg = f"Invalid Runtime Qualification expires_at: {value!r}"
        raise ValueError(msg) from exc


def _require_profile_value[T](value: T | None, field_name: str) -> T:
    if value is None:
        msg = f"Runtime Qualification requires Cluster Profile field: {field_name}"
        raise RuntimeQualificationConfigError(msg)
    return value


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = result.stderr.strip() or f"git {' '.join(args)} failed"
        raise RuntimeQualificationSourceError("source-unavailable", msg)
    return result.stdout.strip()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp_path.replace(path)


__all__ = [
    "RuntimeQualificationCheck",
    "RuntimeQualificationRecord",
    "check_runtime_qualification",
    "qualify_runtime",
    "render_runtime_qualification_check_yaml",
    "render_runtime_qualification_smoke_script",
    "render_runtime_qualification_yaml",
    "verify_runtime_qualification_snapshot_path",
]
