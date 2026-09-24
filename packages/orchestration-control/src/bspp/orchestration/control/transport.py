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

"""Lightweight Control Plane command transport helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import posixpath
import re
import secrets
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol

from bspp.orchestration.control.monitoring import (
    SlurmCommandSnapshot,
    SlurmJobRecord,
    SlurmJobState,
    SlurmObservation,
    parse_parsable_state_rows,
    parse_sacct_identity_parsable_rows,
    parse_sacct_json,
    parse_sacct_parsable_rows,
    parse_squeue_json,
    selected_records_by_job_id,
)

TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)
_PARSABLE_JOB_ID_RE = re.compile(r"^\s*(?P<job_id>\d+)(?:;[^\s]+)?\s*$")
_SUBMITTED_JOB_ID_RE = re.compile(r"\bSubmitted\s+batch\s+job\s+(?P<job_id>\d+)\b")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STAGING_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]+")
_NUMERIC_JOB_ID_RE = re.compile(r"[0-9]+")


_REMOTE_ARTIFACT_PROGRAM = r"""
import base64, hashlib, json, os, stat, sys

MISSING = 44

def fail(message, code=1):
    print(message, file=sys.stderr)
    raise SystemExit(code)

def checked_relative(value):
    parts = value.split('/')
    if not value or value.startswith('/') or any(part in ('', '.', '..') for part in parts):
        fail('BSPP_UNSAFE: invalid relative path')
    return parts

def open_root(root, create=False):
    if create:
        os.makedirs(root, mode=0o700, exist_ok=True)
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        fail('BSPP_MISSING: remote evidence root', MISSING)
    except OSError:
        fail('BSPP_UNSAFE: remote evidence root')
    current = os.fstat(fd)
    if not stat.S_ISDIR(current.st_mode):
        os.close(fd)
        fail('BSPP_UNSAFE: remote evidence root')
    return fd

def walk(root_fd, parts, create=False):
    fd = os.dup(root_fd)
    try:
        for name in parts:
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                fail('BSPP_MISSING: remote artifact parent', MISSING)
            except OSError:
                fail('BSPP_UNSAFE: remote artifact parent')
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                fail('BSPP_UNSAFE: remote artifact parent')
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise

def revalidate(root_fd, parts, held_fd):
    reached = walk(root_fd, parts)
    try:
        held = os.fstat(held_fd); current = os.fstat(reached)
        if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            fail('BSPP_UNSAFE: remote artifact directory replaced during operation')
    finally:
        os.close(reached)

def revalidate_root(root_path, held_fd):
    try:
        reached = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        fail('BSPP_UNSAFE: remote evidence root replaced during operation')
    try:
        held = os.fstat(held_fd); current = os.fstat(reached)
        if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            fail('BSPP_UNSAFE: remote evidence root replaced during operation')
    finally:
        os.close(reached)

def open_regular(parent_fd, name, missing_ok=False):
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except FileNotFoundError:
        if missing_ok:
            return None
        fail('BSPP_MISSING: remote artifact', MISSING)
    except OSError:
        fail('BSPP_UNSAFE: remote artifact path')
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        fail('BSPP_UNSAFE: remote artifact path')
    return fd

def identity(fd, maximum=None):
    info = os.fstat(fd)
    if maximum is not None and info.st_size > maximum:
        fail('BSPP_UNSAFE: remote artifact exceeds size bound')
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        block = os.read(fd, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    after = os.fstat(fd)
    before_identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_nlink)
    if before_identity != after_identity:
        fail('BSPP_UNSAFE: remote artifact changed during read')
    return info.st_size, digest.hexdigest()

args = json.loads(sys.argv[1])
operation = args['operation']
root_fd = open_root(args['root'], create=operation in ('prepare-stage', 'publish-input'))
try:
    relative = checked_relative(args['relative'])
    transfer = checked_relative(args['transfer'])
    if operation == 'prepare-stage':
        parent = walk(root_fd, relative[:-1], create=True)
        os.close(parent)
        transfer_parent = walk(root_fd, transfer[:-1], create=True)
        try:
            os.mkdir(transfer[-1], mode=0o700, dir_fd=transfer_parent)
        except FileExistsError:
            fail('BSPP_UNSAFE: remote transfer directory already exists')
        transfer_fd = os.open(transfer[-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=transfer_parent)
        os.close(transfer_parent)
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            payload = os.open('payload', flags, 0o600, dir_fd=transfer_fd)
            os.close(payload)
            os.fsync(transfer_fd)
            revalidate(root_fd, transfer, transfer_fd)
        finally:
            os.close(transfer_fd)
        print(json.dumps({'temporary': args['root'].rstrip('/') + '/' + args['transfer'] + '/payload'}))
    elif operation == 'publish-input':
        data = sys.stdin.buffer.read(args['size'] + 1)
        if len(data) != args['size'] or hashlib.sha256(data).hexdigest() != args['sha256']:
            fail('BSPP_UNSAFE: streamed artifact identity mismatch')
        parent = walk(root_fd, relative[:-1], create=True)
        existing = open_regular(parent, relative[-1], missing_ok=True)
        if existing is None:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            target = os.open(relative[-1], flags, 0o600, dir_fd=parent)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(target, view)
                    if written <= 0:
                        fail('BSPP_UNSAFE: stalled remote artifact write')
                    view = view[written:]
                os.fsync(target)
            finally:
                os.close(target)
        else:
            existing_identity = identity(existing)
            os.close(existing)
            if existing_identity != (args['size'], args['sha256']):
                fail('BSPP_UNSAFE: immutable remote artifact differs')
        os.fsync(parent)
        revalidate(root_fd, relative[:-1], parent)
        final = open_regular(parent, relative[-1])
        if identity(final) != (args['size'], args['sha256']):
            fail('BSPP_UNSAFE: published remote artifact identity changed')
        os.close(final)
        os.close(parent)
    elif operation == 'publish-stage':
        transfer_parent = walk(root_fd, transfer[:-1])
        transfer_fd = os.open(transfer[-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=transfer_parent)
        os.close(transfer_parent)
        payload = open_regular(transfer_fd, 'payload')
        size, digest = identity(payload)
        if size != args['size'] or digest != args['sha256']:
            fail('BSPP_UNSAFE: staged artifact identity mismatch')
        os.close(payload)
        parent = walk(root_fd, relative[:-1])
        existing = open_regular(parent, relative[-1], missing_ok=True)
        if existing is None:
            os.link('payload', relative[-1], src_dir_fd=transfer_fd, dst_dir_fd=parent, follow_symlinks=False)
            os.unlink('payload', dir_fd=transfer_fd)
        else:
            existing_identity = identity(existing)
            os.close(existing)
            if existing_identity != (size, digest):
                fail('BSPP_UNSAFE: immutable remote artifact differs')
            os.unlink('payload', dir_fd=transfer_fd)
        os.fsync(parent)
        revalidate(root_fd, relative[:-1], parent)
        final = open_regular(parent, relative[-1])
        if identity(final) != (size, digest):
            fail('BSPP_UNSAFE: published remote artifact identity changed')
        os.close(final)
        os.close(parent)
        revalidate(root_fd, transfer, transfer_fd)
        os.close(transfer_fd)
        transfer_parent = walk(root_fd, transfer[:-1])
        os.rmdir(transfer[-1], dir_fd=transfer_parent)
        os.fsync(transfer_parent)
        os.close(transfer_parent)
    elif operation == 'prepare-fetch':
        parent = walk(root_fd, relative[:-1])
        source = open_regular(parent, relative[-1])
        size, digest = identity(source, args['maximum'])
        transfer_parent = walk(root_fd, transfer[:-1], create=True)
        try:
            os.mkdir(transfer[-1], mode=0o700, dir_fd=transfer_parent)
        except FileExistsError:
            fail('BSPP_UNSAFE: remote transfer directory already exists')
        transfer_fd = os.open(transfer[-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=transfer_parent)
        os.close(transfer_parent)
        snapshot = os.open('payload', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=transfer_fd)
        try:
            os.lseek(source, 0, os.SEEK_SET)
            while True:
                block = os.read(source, 1024 * 1024)
                if not block:
                    break
                os.write(snapshot, block)
            os.fsync(snapshot)
            if identity(source, args['maximum']) != (size, digest):
                fail('BSPP_UNSAFE: remote artifact changed during snapshot')
            revalidate(root_fd, relative[:-1], parent)
        finally:
            os.close(snapshot)
            os.close(source)
            os.close(parent)
        snapshot = open_regular(transfer_fd, 'payload')
        if identity(snapshot) != (size, digest):
            fail('BSPP_UNSAFE: remote snapshot identity mismatch')
        os.close(snapshot)
        os.close(transfer_fd)
        temporary = args['root'].rstrip('/') + '/' + args['transfer'] + '/payload'
        print(json.dumps({'temporary': temporary, 'size': size, 'sha256': digest}))
    elif operation == 'fetch-output':
        parent = walk(root_fd, relative[:-1])
        source = open_regular(parent, relative[-1])
        size, digest = identity(source, args['maximum'])
        os.lseek(source, 0, os.SEEK_SET)
        chunks = []
        while True:
            block = os.read(source, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        if identity(source, args['maximum']) != (size, digest):
            fail('BSPP_UNSAFE: remote artifact changed during fetch')
        revalidate(root_fd, relative[:-1], parent)
        final = open_regular(parent, relative[-1])
        if identity(final, args['maximum']) != (size, digest):
            fail('BSPP_UNSAFE: remote artifact identity changed during fetch')
        os.close(final); os.close(source); os.close(parent)
        print(json.dumps({'size': size, 'sha256': digest, 'data': base64.b64encode(b''.join(chunks)).decode()}))
    elif operation == 'finish-fetch':
        transfer_parent = walk(root_fd, transfer[:-1])
        transfer_fd = os.open(transfer[-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=transfer_parent)
        payload = open_regular(transfer_fd, 'payload')
        size, digest = identity(payload, args['maximum'])
        os.close(payload)
        if size != args['size'] or digest != args['sha256']:
            fail('BSPP_UNSAFE: remote snapshot changed during fetch')
        os.unlink('payload', dir_fd=transfer_fd)
        revalidate(root_fd, transfer, transfer_fd)
        os.close(transfer_fd)
        os.rmdir(transfer[-1], dir_fd=transfer_parent)
        os.fsync(transfer_parent)
        os.close(transfer_parent)
    elif operation == 'ensure-directory':
        directory = walk(root_fd, relative, create=True)
        os.fsync(directory)
        revalidate(root_fd, relative, directory)
        os.close(directory)
    else:
        fail('BSPP_UNSAFE: unknown remote artifact operation')
finally:
    revalidate_root(args['root'], root_fd)
    os.close(root_fd)
"""


@dataclass(frozen=True)
class CommandResult:
    """Completed command result captured by Control Plane probes."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Callable command runner used by dry-run probes."""

    def __call__(self, argv: tuple[str, ...]) -> CommandResult:
        """Run one command and return captured output."""


def default_command_runner(argv: tuple[str, ...]) -> CommandResult:
    """Run a local command without shell expansion."""
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    return CommandResult(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _default_command_runner_input(argv: tuple[str, ...], data: bytes) -> CommandResult:
    result = subprocess.run(argv, input=data, capture_output=True, check=False)
    return CommandResult(
        argv=argv,
        returncode=result.returncode,
        stdout=result.stdout.decode(errors="replace"),
        stderr=result.stderr.decode(errors="replace"),
    )


@dataclass(frozen=True)
class SlurmAction:
    """One already-rendered, phase-neutral action submitted to Slurm."""

    action_id: str
    script_path: Path
    dependency_job_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlurmSubmission:
    """Result of a Slurm submission through a Control Plane transport."""

    job_id: str
    command: tuple[str, ...]
    result: CommandResult


class SlurmSubmissionRejected(ValueError):  # noqa: N818 - accepted domain event terminology
    """A local ``sbatch`` process definitively rejected a submission."""

    def __init__(self, message: str, *, result: CommandResult) -> None:
        super().__init__(message)
        self.result = result


class SlurmSubmissionUncertain(ValueError):  # noqa: N818 - accepted domain event terminology
    """A returned transport result does not prove whether Slurm accepted work."""

    def __init__(self, message: str, *, result: CommandResult) -> None:
        super().__init__(message)
        self.result = result


class SlurmActionTransport(Protocol):
    """Scheduler-effects boundary for phase-neutral Slurm actions."""

    def submit_action(self, action: SlurmAction) -> SlurmSubmission:
        """Submit one action and return its scheduler identity."""

    def query_observation(
        self,
        job_ids: tuple[str, ...],
        *,
        require_exact_terminal_exit: bool = False,
        expected_terminal_job_ids: tuple[str, ...] = (),
    ) -> SlurmObservation:
        """Observe the requested scheduler jobs."""

    def query_observation_best_effort(
        self,
        job_ids: tuple[str, ...],
        *,
        require_exact_terminal_exit: bool = False,
        expected_terminal_job_ids: tuple[str, ...] = (),
    ) -> SlurmObservation:
        """Observe jobs while retaining independent scheduler-source failures."""

    def cancel_jobs(self, job_ids: tuple[str, ...]) -> tuple[CommandResult, ...]:
        """Cancel exactly the requested scheduler jobs."""

    def request_job_cancellation(self, job_id: str) -> CommandResult:
        """Request cancellation for exactly one numeric scheduler job."""


@dataclass(frozen=True)
class RemoteSlurmTransport:
    """Explicit local or SSH command transport for Slurm operations."""

    kind: str
    ssh_target: str | None
    runner: CommandRunner = default_command_runner

    def __post_init__(self) -> None:
        _validate_transport(self.kind, self.ssh_target)

    def command(self, argv: tuple[str, ...]) -> CommandResult:
        """Run one command through the selected transport."""
        return self.runner(command_argv(argv, transport=self.kind, ssh_target=self.ssh_target))

    def shell(self, script: str) -> CommandResult:
        """Run one shell script through the selected transport."""
        return self.runner(shell_argv(script, transport=self.kind, ssh_target=self.ssh_target))

    def copy_artifact(self, local_path: Path, target_path: str) -> CommandResult:
        """Copy one local artifact to the transport target."""
        return self.runner(artifact_copy_argv(local_path, target_path, transport=self.kind, ssh_target=self.ssh_target))

    def stage_immutable_artifact(
        self,
        local_path: Path,
        target_path: str,
        *,
        expected_sha256: str,
        staging_token: str,
    ) -> None:
        """Publish exact bytes to one cluster path without replacing divergent content."""
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError("immutable artifact expected_sha256 must be lowercase SHA-256")
        if _STAGING_TOKEN_RE.fullmatch(staging_token) is None:
            raise ValueError("immutable artifact staging token contains unsupported characters")
        if not local_path.is_file() or local_path.is_symlink():
            raise ValueError(f"immutable artifact source must be a regular file: {local_path}")
        if not target_path.startswith("/"):
            raise ValueError("immutable artifact target must be an absolute cluster path")
        parent = posixpath.dirname(target_path)
        temporary = f"{target_path}.bspp-stage-{staging_token}.tmp"
        _require_command_success(self.command(("mkdir", "-p", parent)), operation="artifact staging mkdir")
        _require_command_success(self.copy_artifact(local_path, temporary), operation="artifact staging copy")
        observed = _sha256_from_result(
            self.command(("sha256sum", temporary)),
            operation="artifact staging verification",
        )
        if observed != expected_sha256:
            raise ValueError(
                f"staged immutable artifact SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
            )
        publish_script = "\n".join(
            (
                "set -euo pipefail",
                f"temporary={shlex.quote(temporary)}",
                f"target={shlex.quote(target_path)}",
                f"expected={shlex.quote(expected_sha256)}",
                'if [[ -e "$target" ]]; then',
                '  observed="$(sha256sum "$target" | awk \'{print $1}\')"',
                '  [[ "$observed" == "$expected" ]] || { echo "immutable artifact collision" >&2; exit 73; }',
                '  rm -f -- "$temporary"',
                "else",
                '  ln -- "$temporary" "$target"',
                '  rm -f -- "$temporary"',
                "fi",
            )
        )
        _require_command_success(self.shell(publish_script), operation="immutable artifact publication")

    def replace_artifact_atomically(
        self,
        local_path: Path,
        target_path: str,
        *,
        expected_sha256: str,
        staging_token: str,
        lock_path: str,
    ) -> None:
        """Replace one mutable authority file under its transport-side lock."""
        if not lock_path.startswith("/") or "\x00" in lock_path:
            raise ValueError("mutable artifact lock must be an absolute cluster path")
        temporary = f"{target_path}.bspp-replace-{staging_token}.tmp"
        self.stage_immutable_artifact(
            local_path,
            temporary,
            expected_sha256=expected_sha256,
            staging_token=staging_token,
        )
        fsync_program = "\n".join(
            (
                "import os, sys",
                "descriptor = os.open(sys.argv[1], os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))",
                "try:",
                "    os.fsync(descriptor)",
                "finally:",
                "    os.close(descriptor)",
            )
        )
        replace_script = "\n".join(
            (
                "set -euo pipefail",
                f"temporary={shlex.quote(temporary)}",
                f"target={shlex.quote(target_path)}",
                f"expected={shlex.quote(expected_sha256)}",
                f"lock={shlex.quote(lock_path)}",
                'exec 9>>"$lock"',
                "flock -x 9",
                'observed="$(sha256sum "$temporary" | awk \'{print $1}\')"',
                '[[ "$observed" == "$expected" ]] || { echo "mutable artifact collision" >&2; exit 73; }',
                f'python3 -c {shlex.quote(fsync_program)} "$temporary"',
                'mv -f -- "$temporary" "$target"',
                'parent="$(dirname -- "$target")"',
                f'python3 -c {shlex.quote(fsync_program)} "$parent"',
            )
        )
        _require_command_success(self.shell(replace_script), operation="mutable artifact replacement")

    def read_immutable_bytes_artifact_no_follow(
        self,
        target_path: str,
        *,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> bytes:
        """Read bounded opaque bytes without following the final path symlink."""
        if not target_path.startswith("/") or "\x00" in target_path:
            raise ValueError("immutable artifact path must be absolute")
        if max_bytes <= 0:
            raise ValueError("immutable artifact max_bytes must be positive")
        program = "\n".join(
            (
                "import base64, os, stat, sys",
                "path = sys.argv[1]",
                "limit = int(sys.argv[2])",
                "fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))",
                "try:",
                "    info = os.fstat(fd)",
                "    if not stat.S_ISREG(info.st_mode):",
                "        raise ValueError('artifact is not a regular file')",
                "    if info.st_size > limit:",
                "        raise ValueError('artifact exceeds bounded read limit')",
                "    chunks = []",
                "    remaining = limit + 1",
                "    while remaining:",
                "        chunk = os.read(fd, min(1024 * 1024, remaining))",
                "        if not chunk:",
                "            break",
                "        chunks.append(chunk)",
                "        remaining -= len(chunk)",
                "    data = b''.join(chunks)",
                "    if len(data) > limit:",
                "        raise ValueError('artifact exceeds bounded read limit')",
                "finally:",
                "    os.close(fd)",
                "sys.stdout.write(base64.b64encode(data).decode('ascii'))",
            )
        )
        result = self.command(("python3", "-c", program, target_path, str(max_bytes)))
        _require_command_success(result, operation="immutable text artifact read")
        try:
            data = base64.b64decode(result.stdout, validate=True)
        except ValueError as exc:
            raise ValueError("immutable artifact read returned invalid encoded bytes") from exc
        if len(data) > max_bytes:
            raise ValueError("immutable artifact exceeds bounded read limit")
        return data

    def read_immutable_text_artifact_no_follow(
        self,
        target_path: str,
        *,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> bytes:
        """Read one bounded UTF-8 regular file without following its final symlink."""
        data = self.read_immutable_bytes_artifact_no_follow(target_path, max_bytes=max_bytes)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("immutable text artifact must be UTF-8") from exc
        return data

    def fetch_artifact(self, remote_path: str, local_path: Path) -> CommandResult:
        """Fetch one exact artifact from the transport target."""
        return self.runner(
            artifact_fetch_argv(remote_path, local_path, transport=self.kind, ssh_target=self.ssh_target)
        )

    def _remote_artifact_operation(self, arguments: dict[str, object]) -> CommandResult:
        return self.command(("python3", "-c", _REMOTE_ARTIFACT_PROGRAM, json.dumps(arguments, sort_keys=True)))

    def _remote_artifact_operation_input(self, arguments: dict[str, object], data: bytes) -> CommandResult:
        argv = command_argv(
            ("python3", "-c", _REMOTE_ARTIFACT_PROGRAM, json.dumps(arguments, sort_keys=True)),
            transport=self.kind,
            ssh_target=self.ssh_target,
        )
        input_runner = getattr(self.runner, "run_with_input", None)
        if callable(input_runner):
            result = input_runner(argv, data)
            if not isinstance(result, CommandResult):
                raise TypeError("streaming command runner returned an invalid result")
            return result
        if self.runner is default_command_runner:
            return _default_command_runner_input(argv, data)
        raise ValueError("command runner does not support secure streamed artifact staging")

    @staticmethod
    def _remote_relative(remote_path: Path, remote_root: Path) -> str:
        try:
            relative = remote_path.relative_to(remote_root)
        except ValueError as exc:
            raise ValueError("remote artifact must be below the governed evidence root") from exc
        rendered = PurePosixPath(*relative.parts).as_posix()
        if rendered in {"", "."} or any(part in {"", ".", ".."} for part in PurePosixPath(rendered).parts):
            raise ValueError("remote artifact must be a strict descendant of the governed evidence root")
        return rendered

    @staticmethod
    def _raise_remote_artifact_error(result: CommandResult, *, missing_allowed: bool = False) -> None:
        detail = result.stderr.strip() or result.stdout.strip()
        if result.returncode == 44 and missing_allowed:
            raise FileNotFoundError(detail or "remote artifact is missing")
        raise ValueError("unsafe remote artifact path" + (f": {detail}" if detail else ""))

    def ensure_remote_directory(self, remote_root: Path, relative: Path) -> None:
        """Create a root-confined remote directory without following descendants."""
        rendered = self._remote_relative(remote_root / relative, remote_root)
        transfer = f".bspp-transfer/{secrets.token_hex(16)}"
        result = self._remote_artifact_operation(
            {"operation": "ensure-directory", "root": str(remote_root), "relative": rendered, "transfer": transfer}
        )
        if result.returncode != 0:
            self._raise_remote_artifact_error(result)

    def stage_verified_artifact(
        self,
        local_path: Path,
        remote_path: Path,
        *,
        remote_root: Path,
        expected_sha256: str,
    ) -> None:
        """Publish exact local bytes remotely through a verified create-once path."""
        data = local_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256:
            raise ValueError("local artifact changed after its governed identity was bound")
        relative = self._remote_relative(remote_path, remote_root)
        published = self._remote_artifact_operation_input(
            {
                "operation": "publish-input",
                "root": str(remote_root),
                "relative": relative,
                "transfer": ".bspp-transfer/unused",
                "size": len(data),
                "sha256": digest,
            },
            data,
        )
        if published.returncode != 0:
            self._raise_remote_artifact_error(published)

    def fetch_stable_artifact(self, remote_path: Path, *, remote_root: Path, maximum_bytes: int) -> bytes:
        """Fetch bytes only when the remote regular-file identity is stable."""
        relative = self._remote_relative(remote_path, remote_root)
        fetched = self._remote_artifact_operation(
            {
                "operation": "fetch-output",
                "root": str(remote_root),
                "relative": relative,
                "transfer": ".bspp-transfer/unused",
                "maximum": maximum_bytes,
            }
        )
        if fetched.returncode != 0:
            self._raise_remote_artifact_error(fetched, missing_allowed=True)
        try:
            response = json.loads(fetched.stdout)
            size = int(response["size"])
            digest = str(response["sha256"])
            data = base64.b64decode(response["data"], validate=True)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("malformed remote artifact fetch response") from exc
        if size > maximum_bytes or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("malformed remote artifact fetch identity")
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("fetched artifact does not match remote identity")
        return data

    def submit_script(
        self,
        script_path: Path,
        *,
        dependencies: tuple[str, ...] = (),
        job_name: str | None = None,
        environment: tuple[tuple[str, str], ...] = (),
    ) -> SlurmSubmission:
        """Submit a rendered sbatch script and return the parsed job id."""
        return self._submit_sbatch(
            script_path,
            dependencies=dependencies,
            job_name=job_name,
            environment=environment,
        )

    def submit_action(self, action: SlurmAction) -> SlurmSubmission:
        """Submit one rendered action; its id is caller-association metadata only."""
        return self._submit_sbatch(action.script_path, dependencies=action.dependency_job_ids)

    def _submit_sbatch(
        self,
        script_path: Path,
        *,
        dependencies: tuple[str, ...],
        job_name: str | None = None,
        environment: tuple[tuple[str, str], ...] = (),
    ) -> SlurmSubmission:
        """Build and submit one sbatch command from explicit scheduler-facing fields."""
        command: tuple[str, ...] = ("sbatch", "--parsable")
        if dependencies:
            command = (*command, "--dependency=afterok:" + ":".join(dependencies))
        if job_name is not None:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_name):
                raise ValueError("invalid Slurm job name")
            command = (*command, f"--job-name={job_name}")
        if environment:
            rendered_environment: list[str] = []
            for name, value in environment:
                if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", value):
                    raise ValueError("invalid Slurm submission environment binding")
                rendered_environment.append(f"{name}={value}")
            command = (*command, "--export=ALL," + ",".join(rendered_environment))
        result = self.command((*command, str(script_path)))
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "sbatch submission failed"
            failure = SlurmSubmissionRejected if self.kind == "local-slurm" else SlurmSubmissionUncertain
            raise failure(detail, result=result)
        try:
            job_id = parse_sbatch_job_id(result.stdout)
        except ValueError as exc:
            raise SlurmSubmissionUncertain(str(exc), result=result) from exc
        return SlurmSubmission(
            job_id=job_id,
            command=result.argv,
            result=result,
        )

    def query_submissions_by_correlation(
        self,
        *,
        job_name: str,
        comment: str,
        submitted_after: str,
    ) -> tuple[SlurmJobRecord, ...]:
        """Find top-level numeric jobs with the exact durable name/comment token."""
        if not job_name or not comment or not submitted_after:
            raise ValueError("submission correlation requires job name, comment, and lower-bound timestamp")
        records = (
            *self._query_squeue_correlation(job_name),
            *self._query_sacct_correlation(job_name, submitted_after=submitted_after),
        )
        by_id: dict[str, SlurmJobRecord] = {}
        for record in records:
            if (
                _NUMERIC_JOB_ID_RE.fullmatch(record.job_id) is None
                or record.name != job_name
                or record.comment != comment
            ):
                continue
            existing = by_id.get(record.job_id)
            if existing is None or (record.source == "sacct" and existing.source != "sacct"):
                by_id[record.job_id] = record
        return tuple(by_id[job_id] for job_id in sorted(by_id, key=int))

    def _query_squeue_correlation(self, job_name: str) -> tuple[SlurmJobRecord, ...]:
        result = self.command(("squeue", "--json", f"--name={job_name}"))
        if result.returncode == 0:
            try:
                return parse_squeue_json(result.stdout, requested=())
            except ValueError:
                pass
        fallback = self.command(("squeue", "-h", f"--name={job_name}", "-o", "%i|%j|%k|%T|%V"))
        return _parse_correlation_rows(fallback, source="squeue")

    def _query_sacct_correlation(
        self,
        job_name: str,
        *,
        submitted_after: str,
    ) -> tuple[SlurmJobRecord, ...]:
        slurm_starttime = _slurm_starttime(submitted_after)
        result = self.command(
            (
                "sacct",
                "--json",
                "-X",
                f"--name={job_name}",
                f"--starttime={slurm_starttime}",
                "--format=JobIDRaw,JobName,Comment,State,ExitCode,Submit",
            )
        )
        if result.returncode == 0:
            try:
                return parse_sacct_json(result.stdout, requested=())
            except ValueError:
                pass
        fallback = self.command(
            (
                "sacct",
                "-X",
                f"--name={job_name}",
                f"--starttime={slurm_starttime}",
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,JobName,Comment,State,Submit",
            )
        )
        return _parse_correlation_rows(fallback, source="sacct")

    def find_governed_submission_job(self, *, submission_token: str, owner: str) -> str | None:
        """Find one live or completed job bearing an exact governed submission token."""
        if not re.fullmatch(r"[0-9a-f]{64}", submission_token):
            raise ValueError("invalid governed submission token")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
            raise ValueError("invalid governed submission owner")
        job_name = f"bspp_sub_{submission_token}"
        result = self.command(("squeue", "--noheader", f"--user={owner}", f"--name={job_name}", "--format=%A"))
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or "governed submission reconciliation failed")
        jobs = tuple(dict.fromkeys(line.strip() for line in result.stdout.splitlines() if line.strip()))
        if any(not job.isdecimal() for job in jobs) or len(jobs) > 1:
            raise ValueError("ambiguous governed submission scheduler reconciliation")
        if jobs:
            return jobs[0]
        accounting = self.command(
            (
                "sacct",
                "-X",
                "--noheader",
                f"--user={owner}",
                f"--name={job_name}",
                "--format=JobIDRaw,JobName,User,State",
                "--parsable2",
            )
        )
        if accounting.returncode != 0:
            raise ValueError(accounting.stderr.strip() or "governed submission accounting reconciliation failed")
        accounted: list[str] = []
        for line in accounting.stdout.splitlines():
            fields = line.strip().split("|")
            if fields and fields[-1] == "":
                fields.pop()
            if len(fields) != 4 or fields[1] != job_name or fields[2] != owner or not fields[3].strip():
                raise ValueError("malformed governed submission accounting reconciliation output")
            base_job_id = fields[0].split("_", 1)[0]
            if not base_job_id.isdecimal() or not re.fullmatch(r"\d+(?:_\d+)?", fields[0]):
                raise ValueError("malformed governed submission accounting base job row")
            accounted.append(base_job_id)
        unique = tuple(dict.fromkeys(accounted))
        if len(unique) > 1:
            raise ValueError("ambiguous governed submission accounting reconciliation")
        return unique[0] if unique else None

    def find_runtime_qualification_job(self, *, attempt_token: str, profile: str, owner: str) -> str | None:
        """Find the unique scheduler job bearing an exact Runtime Qualification attempt name."""
        if not re.fullmatch(r"[0-9a-f]{32}", attempt_token):
            raise ValueError("invalid Runtime Qualification attempt token")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", profile) or not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
            raise ValueError("invalid Runtime Qualification scheduler selector")
        result = self.command(
            (
                "squeue",
                "--noheader",
                f"--user={owner}",
                f"--name=bspp_rq_{profile}_{attempt_token}",
                "--format=%A",
            )
        )
        if result.returncode != 0:
            raise ValueError(result.stderr.strip() or "Runtime Qualification reconciliation failed")
        jobs = tuple(dict.fromkeys(line.strip() for line in result.stdout.splitlines() if line.strip()))
        if len(jobs) > 1:
            raise ValueError("ambiguous Runtime Qualification scheduler reconciliation")
        if jobs:
            return jobs[0]
        job_name = f"bspp_rq_{profile}_{attempt_token}"
        accounting = self.command(
            (
                "sacct",
                "-X",
                "--noheader",
                f"--user={owner}",
                f"--name={job_name}",
                "--format=JobIDRaw,JobName,User,State",
                "--parsable2",
            )
        )
        if accounting.returncode != 0:
            raise ValueError(accounting.stderr.strip() or "Runtime Qualification accounting reconciliation failed")
        accounted: list[str] = []
        for line in accounting.stdout.splitlines():
            rendered = line.strip()
            if not rendered:
                continue
            fields = rendered.split("|")
            if fields and fields[-1] == "":
                fields.pop()
            if len(fields) != 4 or fields[1] != job_name or fields[2] != owner or not fields[3].strip():
                raise ValueError("malformed Runtime Qualification accounting reconciliation output")
            job_id = fields[0]
            if not job_id.isdecimal():
                raise ValueError("Runtime Qualification accounting returned non-base job row")
            accounted.append(job_id)
        unique = tuple(dict.fromkeys(accounted))
        if len(unique) > 1:
            raise ValueError("ambiguous Runtime Qualification accounting reconciliation")
        return unique[0] if unique else None

    def query_job_states(self, job_ids: tuple[str, ...]) -> tuple[SlurmJobState, ...]:
        """Inspect squeue and sacct for normalized states keyed by requested job id."""
        return self.query_observation(job_ids).selected_states

    def query_observation(
        self,
        job_ids: tuple[str, ...],
        *,
        require_exact_terminal_exit: bool = False,
        expected_terminal_job_ids: tuple[str, ...] = (),
    ) -> SlurmObservation:
        """Inspect squeue and sacct and return a structured scheduler observation."""
        return self._query_observation(
            job_ids,
            best_effort=False,
            require_exact_terminal_exit=require_exact_terminal_exit,
            expected_terminal_job_ids=expected_terminal_job_ids,
        )

    def query_observation_best_effort(
        self,
        job_ids: tuple[str, ...],
        *,
        require_exact_terminal_exit: bool = False,
        expected_terminal_job_ids: tuple[str, ...] = (),
    ) -> SlurmObservation:
        """Inspect both scheduler sources while retaining source unavailability."""
        return self._query_observation(
            job_ids,
            best_effort=True,
            require_exact_terminal_exit=require_exact_terminal_exit,
            expected_terminal_job_ids=expected_terminal_job_ids,
        )

    def _query_observation(
        self,
        job_ids: tuple[str, ...],
        *,
        best_effort: bool,
        require_exact_terminal_exit: bool,
        expected_terminal_job_ids: tuple[str, ...],
    ) -> SlurmObservation:
        requested = _unique_job_ids(job_ids)
        expected_terminal_job_ids = _validate_expected_terminal_job_ids(
            requested,
            expected_terminal_job_ids,
            require_exact_terminal_exit=require_exact_terminal_exit,
        )
        if not requested:
            empty_squeue = SlurmCommandSnapshot(kind="squeue", argv=(), returncode=0, parser="skipped")
            empty_sacct = SlurmCommandSnapshot(kind="sacct", argv=(), returncode=0, parser="skipped")
            return SlurmObservation(
                requested_job_ids=(),
                squeue=empty_squeue,
                sacct=empty_sacct,
                squeue_jobs=(),
                sacct_jobs=(),
                selected_states=(),
            )
        squeue_snapshot, squeue_jobs, squeue_warnings = self._query_squeue(
            requested,
            best_effort=best_effort,
            scoped=require_exact_terminal_exit,
        )
        sacct_snapshot, sacct_jobs, sacct_warnings = self._query_sacct(
            requested,
            best_effort=best_effort,
            require_exact_terminal_exit=require_exact_terminal_exit,
            expected_terminal_job_ids=expected_terminal_job_ids,
        )
        return SlurmObservation(
            requested_job_ids=requested,
            squeue=squeue_snapshot,
            sacct=sacct_snapshot,
            squeue_jobs=squeue_jobs,
            sacct_jobs=sacct_jobs,
            selected_states=_select_job_states(requested, squeue_jobs=squeue_jobs, sacct_jobs=sacct_jobs),
            warnings=(*squeue_warnings, *sacct_warnings),
        )

    def cancel_jobs(self, job_ids: tuple[str, ...]) -> tuple[CommandResult, ...]:
        """Cancel exactly the provided job ids."""
        results: list[CommandResult] = []
        for job_id in _unique_job_ids(job_ids):
            result = self.request_job_cancellation(job_id)
            if result.returncode != 0:
                msg = result.stderr.strip() or f"scancel failed for job {job_id}"
                raise ValueError(msg)
            results.append(result)
        return tuple(results)

    def request_job_cancellation(self, job_id: str) -> CommandResult:
        """Invoke one exact ``scancel`` and return its result, including rejection."""
        if _NUMERIC_JOB_ID_RE.fullmatch(job_id) is None:
            raise ValueError("scancel requires one numeric Slurm job id")
        return self.command(("scancel", job_id))

    def _query_squeue(
        self,
        job_ids: tuple[str, ...],
        *,
        best_effort: bool = False,
        scoped: bool = False,
    ) -> tuple[SlurmCommandSnapshot, tuple[SlurmJobRecord, ...], tuple[str, ...]]:
        argv = ("squeue", "--json", "-j", ",".join(job_ids))
        try:
            result = self._run_sacct(argv)
        except OSError as exc:
            if not best_effort:
                raise
            return self._unavailable_source("squeue", argv, detail=str(exc))
        if result.returncode != 0:
            warning = result.stderr.strip() or "squeue --json query failed"
            return self._query_squeue_parsable(
                job_ids,
                warning=f"squeue_json_unavailable: {warning}",
                best_effort=best_effort,
            )
        try:
            parsed_jobs = parse_squeue_json(result.stdout, requested=job_ids)
            jobs = _requested_records(parsed_jobs) if scoped else parsed_jobs
            return (
                SlurmCommandSnapshot(
                    kind="squeue",
                    argv=result.argv,
                    returncode=result.returncode,
                    parser="json",
                    stderr=result.stderr,
                    raw_json=_requested_json_payload(jobs) if scoped else json.loads(result.stdout or "{}"),
                ),
                jobs,
                (),
            )
        except ValueError as exc:
            return self._query_squeue_parsable(
                job_ids,
                warning=f"squeue_json_unavailable: {exc}",
                best_effort=best_effort,
            )

    def _query_sacct(
        self,
        job_ids: tuple[str, ...],
        *,
        best_effort: bool = False,
        require_exact_terminal_exit: bool = False,
        expected_terminal_job_ids: tuple[str, ...] = (),
    ) -> tuple[SlurmCommandSnapshot, tuple[SlurmJobRecord, ...], tuple[str, ...]]:
        argv = (
            "sacct",
            "--json",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem,Restarts",
        )
        try:
            result = self._run_sacct(argv)
        except OSError as exc:
            if not best_effort:
                raise
            return self._unavailable_source("sacct", argv, detail=str(exc))
        if result.returncode != 0:
            warning = result.stderr.strip() or "sacct --json accounting query failed"
            return self._query_sacct_parsable(
                job_ids,
                warning=f"sacct_json_unavailable: {warning}",
                best_effort=best_effort,
                require_exact_terminal_exit=require_exact_terminal_exit,
                identity_cross_check=require_exact_terminal_exit,
            )
        try:
            parsed_jobs = parse_sacct_json(
                result.stdout,
                requested=job_ids,
                strict_rows=require_exact_terminal_exit,
            )
        except ValueError as exc:
            return self._query_sacct_parsable(
                job_ids,
                warning=f"sacct_json_unavailable: {exc}",
                best_effort=best_effort,
                require_exact_terminal_exit=require_exact_terminal_exit,
                identity_cross_check=require_exact_terminal_exit,
            )

        if not require_exact_terminal_exit:
            return (
                SlurmCommandSnapshot(
                    kind="sacct",
                    argv=result.argv,
                    returncode=result.returncode,
                    parser="json",
                    stderr=result.stderr,
                    raw_json=json.loads(result.stdout or "{}"),
                ),
                parsed_jobs,
                (),
            )

        jobs = _requested_parent_hierarchy_records(parsed_jobs, requested=job_ids)

        sparse_terminal_jobs = tuple(
            job
            for job in jobs
            if job.state in TERMINAL_SLURM_STATES
            and job.exit_code is None
            and (not expected_terminal_job_ids or job.job_id in expected_terminal_job_ids)
        )
        missing_array_endpoints = tuple(
            endpoint
            for endpoint in expected_terminal_job_ids
            if "_" in endpoint
            and not any(job.job_id == endpoint for job in jobs)
            and any(job.job_id == endpoint.split("_", 1)[0] and job.state in TERMINAL_SLURM_STATES for job in jobs)
        )
        if not sparse_terminal_jobs and not missing_array_endpoints:
            return _json_sacct_result(result, jobs)

        reasons: list[str] = []
        if sparse_terminal_jobs:
            reasons.append("terminal accounting record lacks an exact exit code")
        if missing_array_endpoints:
            reasons.append("terminal parent accounting lacks an expected array child")
        incomplete_warning = "sacct_json_incomplete: " + "; ".join(reasons)
        fallback_snapshot, fallback_jobs, fallback_warnings = self._query_sacct_parsable(
            job_ids,
            warning=incomplete_warning,
            # Sparse JSON remains usable in best-effort mode.  Normal observation
            # mode keeps the existing fail-closed transport contract.
            best_effort=best_effort,
            require_exact_terminal_exit=True,
            identity_cross_check=True,
        )
        if fallback_snapshot.parser == "parsable-fallback" and _has_exact_exit_enrichment(
            sparse_terminal_jobs,
            fallback_jobs,
            missing_expected_job_ids=missing_array_endpoints,
        ):
            return fallback_snapshot, fallback_jobs, fallback_warnings

        sparse_exit_enriched = _has_exact_exit_enrichment(sparse_terminal_jobs, fallback_jobs)
        if (
            fallback_snapshot.parser == "parsable-fallback"
            and sparse_exit_enriched
            and _parent_only_cancelled_absence_closure(
                fallback_jobs,
                missing_array_endpoints=missing_array_endpoints,
            )
        ):
            return fallback_snapshot, fallback_jobs, fallback_warnings

        if fallback_snapshot.parser == "parsable-fallback":
            enrichment_detail = "sacct_parsable_enrichment_incomplete: missing exact exit evidence"
        else:
            enrichment_detail = fallback_snapshot.warning or "sacct_parsable_enrichment_unavailable"
        warning = f"{incomplete_warning}; {enrichment_detail}"
        return _json_sacct_result(result, jobs, warning=warning)

    def _query_squeue_parsable(
        self,
        job_ids: tuple[str, ...],
        *,
        warning: str,
        best_effort: bool = False,
    ) -> tuple[SlurmCommandSnapshot, tuple[SlurmJobRecord, ...], tuple[str, ...]]:
        argv = ("squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T")
        try:
            result = self._run_sacct(argv)
        except OSError as exc:
            if not best_effort:
                raise
            return self._unavailable_source("squeue", argv, detail=str(exc), prior_warning=warning)
        if result.returncode != 0:
            msg = result.stderr.strip() or "squeue query failed"
            if not best_effort:
                raise ValueError(msg)
            return self._unavailable_source("squeue", argv, result=result, detail=msg, prior_warning=warning)
        if best_effort:
            try:
                _validate_parsable_state_rows(result.stdout, source="squeue")
            except ValueError as exc:
                return self._unavailable_source(
                    "squeue",
                    argv,
                    result=result,
                    detail=str(exc),
                    prior_warning=warning,
                )
        return (
            SlurmCommandSnapshot(
                kind="squeue",
                argv=result.argv,
                returncode=result.returncode,
                parser="parsable-fallback",
                stderr=result.stderr,
                raw_text=result.stdout,
                warning=warning,
            ),
            parse_parsable_state_rows(result.stdout, source="squeue", requested=job_ids),
            (warning,),
        )

    def _query_sacct_parsable(
        self,
        job_ids: tuple[str, ...],
        *,
        warning: str,
        best_effort: bool = False,
        require_exact_terminal_exit: bool = False,
        identity_cross_check: bool = False,
    ) -> tuple[SlurmCommandSnapshot, tuple[SlurmJobRecord, ...], tuple[str, ...]]:
        argv = (
            (
                "sacct",
                "-j",
                ",".join(job_ids),
                (
                    "--format=JobIDRaw,JobID,State,ExitCode,Restarts"
                    if identity_cross_check
                    else "--format=JobIDRaw,State,ExitCode,Restarts"
                ),
                "--noheader",
                "--parsable2",
            )
            if require_exact_terminal_exit
            else ("sacct", "-j", ",".join(job_ids), "--format=JobIDRaw,State", "--noheader", "--parsable2")
        )
        try:
            result = self._run_sacct(argv)
        except OSError as exc:
            if not best_effort:
                raise
            return self._unavailable_source("sacct", argv, detail=str(exc), prior_warning=warning)
        if result.returncode != 0:
            msg = result.stderr.strip() or "sacct accounting query failed"
            if not best_effort:
                raise ValueError(msg)
            return self._unavailable_source("sacct", argv, result=result, detail=msg, prior_warning=warning)
        if require_exact_terminal_exit:
            try:
                jobs = _requested_parent_hierarchy_records(
                    (
                        parse_sacct_identity_parsable_rows(result.stdout, requested=job_ids)
                        if identity_cross_check
                        else parse_sacct_parsable_rows(result.stdout, requested=job_ids)
                    ),
                    requested=job_ids,
                )
            except ValueError as exc:
                if not best_effort:
                    raise
                return self._unavailable_source(
                    "sacct",
                    argv,
                    result=result,
                    detail=str(exc),
                    prior_warning=warning,
                )
        else:
            if best_effort:
                try:
                    _validate_parsable_state_rows(result.stdout, source="sacct")
                except ValueError as exc:
                    return self._unavailable_source(
                        "sacct",
                        argv,
                        result=result,
                        detail=str(exc),
                        prior_warning=warning,
                    )
            jobs = parse_parsable_state_rows(result.stdout, source="sacct", requested=job_ids)
        return (
            SlurmCommandSnapshot(
                kind="sacct",
                argv=result.argv,
                returncode=result.returncode,
                parser="parsable-fallback",
                stderr=result.stderr,
                raw_text=result.stdout,
                warning=warning,
            ),
            jobs,
            (warning,),
        )

    def _run_sacct(self, argv: tuple[str, ...]) -> CommandResult:
        """Run sacct, retrying without optional fields the local build rejects.

        Slurm builds before the Restarts field existed (e.g. 23.02) fail the whole
        query with "Invalid field requested", which silently degrades every scheduler
        observation to squeue-only. Retry once with the unsupported field removed; the
        record parser already treats restarts as optional.
        """
        result = self.command(argv)
        stderr = result.stderr or ""
        if result.returncode != 0 and "Invalid field requested" in stderr:
            stripped = tuple(arg.replace(",Restarts", "") if arg.startswith("--format=") else arg for arg in argv)
            if stripped != argv:
                result = self.command(stripped)
        return result

    def _unavailable_source(
        self,
        kind: str,
        argv: tuple[str, ...],
        *,
        detail: str,
        result: CommandResult | None = None,
        prior_warning: str | None = None,
    ) -> tuple[SlurmCommandSnapshot, tuple[SlurmJobRecord, ...], tuple[str, ...]]:
        warning = f"{kind}_unavailable: {detail}"
        if prior_warning is not None:
            warning = f"{warning} ({prior_warning})"
        intended_argv = command_argv(argv, transport=self.kind, ssh_target=self.ssh_target)
        exact_argv = result.argv if result is not None and result.argv else intended_argv
        return (
            SlurmCommandSnapshot(
                kind=kind,
                argv=exact_argv,
                returncode=result.returncode if result is not None else -1,
                parser="unavailable",
                stderr=result.stderr if result is not None else detail,
                raw_text=result.stdout if result is not None else None,
                warning=warning,
            ),
            (),
            (warning,),
        )


def command_argv(
    argv: tuple[str, ...],
    *,
    transport: str,
    ssh_target: str | None,
) -> tuple[str, ...]:
    """Return the local argv for a profile transport."""
    _validate_transport(transport, ssh_target)
    if transport == "local-slurm":
        return argv
    if transport == "ssh":
        assert ssh_target is not None
        return ("ssh", ssh_target, _remote_login_command(" ".join(shlex.quote(part) for part in argv)))
    raise AssertionError("unreachable transport")


# Instant the base64-wrapped SSH login-shell wire shape landed on
# mainline (merge 676c8916, 2026-09-18T05:35:07-07:00).  Events recorded before
# this instant may carry either the legacy or the current wire shape on replay;
# events at or after it must record the current shape, because fresh submission
# paths derive argv exclusively from :func:`command_argv`.
LEGACY_WIRE_SHAPE_CUTOVER_AT = "2026-09-18T12:35:07Z"


def legacy_command_argv(
    argv: tuple[str, ...],
    *,
    transport: str,
    ssh_target: str | None,
) -> tuple[str, ...]:
    """Derive the pre-base64 wire form of the same logical command.

    This function exists **ONLY** to replay sealed pre-MR-!84 acceptance
    evidence whose recorded ``sbatch_argv`` uses the legacy
    ``bash -lc '<quoted command>'`` SSH login-shell form.  It must never be
    used to build new invocations — new submissions always derive their argv
    from :func:`command_argv`.  Callers must accept this shape only for events
    that predate :data:`LEGACY_WIRE_SHAPE_CUTOVER_AT`.

    For ``local-slurm`` the result is identical to ``command_argv`` (argv
    passthrough) because the local transport was never base64-wrapped.  For
    ``ssh`` the result is ``("ssh", ssh_target, "bash -lc " + shlex.quote(inner))``
    where ``inner = " ".join(shlex.quote(part) for part in argv)``.
    """
    _validate_transport(transport, ssh_target)
    if transport == "local-slurm":
        return argv
    if transport == "ssh":
        assert ssh_target is not None
        inner = " ".join(shlex.quote(part) for part in argv)
        return ("ssh", ssh_target, "bash -lc " + shlex.quote(inner))
    raise AssertionError("unreachable transport")


def shell_argv(
    script: str,
    *,
    transport: str,
    ssh_target: str | None,
) -> tuple[str, ...]:
    """Return the local argv for running a shell script through a profile transport."""
    _validate_transport(transport, ssh_target)
    if transport == "local-slurm":
        return ("bash", "-lc", script)
    if transport == "ssh":
        assert ssh_target is not None
        return ("ssh", ssh_target, _remote_login_command(script))
    raise AssertionError("unreachable transport")


def run_probe(
    argv: tuple[str, ...],
    *,
    transport: str,
    ssh_target: str | None,
    runner: Callable[[tuple[str, ...]], CommandResult],
) -> CommandResult:
    """Run one probe through the selected transport."""
    return runner(command_argv(argv, transport=transport, ssh_target=ssh_target))


def _remote_login_command(command: str) -> str:
    # ssh hands this string to the REMOTE LOGIN shell before bash ever runs, and that
    # shell is not guaranteed to be POSIX: some sites run csh login shells, which
    # mangle payloads containing newlines or nested quotes ("Badly placed ()'s"). Base64
    # contains no quotes, newlines or metacharacters, so every login shell parses this
    # identically. The decode happens in a command substitution, so the caller's stdin
    # stays intact for streamed artifact staging.
    encoded = base64.b64encode(command.encode()).decode()
    return "bash -lc 'eval \"$(echo " + encoded + " | base64 -d)\"'"


def artifact_copy_argv(
    local_path: Path,
    target_path: str,
    *,
    transport: str,
    ssh_target: str | None,
) -> tuple[str, ...]:
    """Return the local argv for copying one local artifact to the selected transport target."""
    _validate_transport(transport, ssh_target)
    if transport == "local-slurm":
        return ("cp", str(local_path), target_path)
    if transport == "ssh":
        assert ssh_target is not None
        return ("scp", str(local_path), f"{ssh_target}:{target_path}")
    raise AssertionError("unreachable transport")


def artifact_fetch_argv(
    remote_path: str,
    local_path: Path,
    *,
    transport: str,
    ssh_target: str | None,
) -> tuple[str, ...]:
    """Return argv for fetching one exact transport artifact locally."""
    _validate_transport(transport, ssh_target)
    if transport == "local-slurm":
        return ("cp", remote_path, str(local_path))
    assert ssh_target is not None
    return ("scp", f"{ssh_target}:{remote_path}", str(local_path))


def parse_sbatch_job_id(stdout: str) -> str:
    """Parse common sbatch job-id output formats into a bare Slurm job id."""
    for line in stdout.splitlines():
        parsable = _PARSABLE_JOB_ID_RE.fullmatch(line)
        if parsable is not None:
            return parsable.group("job_id")
        submitted = _SUBMITTED_JOB_ID_RE.search(line)
        if submitted is not None:
            return submitted.group("job_id")
    msg = f"could not parse sbatch job id from output: {stdout.strip()!r}"
    raise ValueError(msg)


def _require_command_success(result: CommandResult, *, operation: str) -> None:
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"{operation} failed"
        raise ValueError(detail)


def _validate_parsable_state_rows(stdout: str, *, source: str) -> None:
    """Reject a successful fallback response containing structurally invalid rows."""
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"{source} parsable fallback returned a malformed row at line {line_number}")


def _sha256_from_result(result: CommandResult, *, operation: str) -> str:
    _require_command_success(result, operation=operation)
    first = result.stdout.strip().split(maxsplit=1)
    if not first or _SHA256_RE.fullmatch(first[0]) is None:
        raise ValueError(f"{operation} did not return a SHA-256 checksum")
    return first[0]


def _parse_correlation_rows(result: CommandResult, *, source: str) -> tuple[SlurmJobRecord, ...]:
    _require_command_success(result, operation=f"{source} submission-correlation query")
    records: list[SlurmJobRecord] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) < 5:
            raise ValueError(f"malformed {source} submission-correlation row")
        job_id, name, comment, state, submitted_at = (field.strip() for field in fields[:5])
        records.append(
            SlurmJobRecord(
                job_id=job_id,
                source=source,
                requested_job_id=None,
                state=state or None,
                name=name or None,
                comment=comment or None,
                submitted_at=submitted_at or None,
                raw=line,
            )
        )
    return tuple(records)


def _slurm_starttime(value: str) -> str:
    """Normalize a durable UTC event timestamp to Slurm's second-resolution grammar."""
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else ""))
    except ValueError as exc:
        raise ValueError("submission correlation lower bound must be a valid timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("submission correlation lower bound must be a valid timezone-aware timestamp")
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def normalize_slurm_state(state: str) -> str:
    """Return the stable Slurm state token used by Control State."""
    return state.strip().upper().split(maxsplit=1)[0] if state.strip() else ""


def is_terminal_slurm_state(state: str | None) -> bool:
    """Return true when a normalized Slurm state is terminal."""
    return normalize_slurm_state(state or "") in TERMINAL_SLURM_STATES


def control_status_from_slurm_state(state: str | None) -> str:
    """Return the terminal Control State status implied by a Slurm state."""
    return "completed" if normalize_slurm_state(state or "") == "COMPLETED" else "failed"


def _validate_transport(transport: str, ssh_target: str | None) -> None:
    if transport == "ssh":
        if ssh_target is None:
            msg = "ssh transport requires ssh_target"
            raise ValueError(msg)
        return
    if transport == "local-slurm":
        return
    msg = f"unsupported transport {transport!r}"
    raise ValueError(msg)


def _unique_job_ids(job_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for job_id in job_ids:
        if job_id in seen:
            continue
        seen.add(job_id)
        unique.append(job_id)
    return tuple(unique)


def _validate_expected_terminal_job_ids(
    requested: tuple[str, ...],
    expected: tuple[str, ...],
    *,
    require_exact_terminal_exit: bool,
) -> tuple[str, ...]:
    """Validate optional exact terminal endpoints against the requested parents."""
    if expected and not require_exact_terminal_exit:
        raise ValueError("expected terminal job ids require exact terminal exit accounting")
    if len(set(expected)) != len(expected):
        raise ValueError("expected terminal job ids must be unique")
    requested_set = set(requested)
    if any(_NUMERIC_JOB_ID_RE.fullmatch(parent) is None for parent in requested):
        raise ValueError("scheduler observation requires numeric parent job ids")
    for endpoint in expected:
        match = re.fullmatch(r"(?P<parent>\d+)(?:_(?P<task>\d+))?", endpoint)
        if match is None or match.group("parent") not in requested_set:
            raise ValueError("expected terminal job id is not an exact requested endpoint")
    return expected


def _requested_records(records: tuple[SlurmJobRecord, ...]) -> tuple[SlurmJobRecord, ...]:
    """Retain only normalized records belonging to the exact requested IDs."""
    return tuple(record for record in records if record.requested_job_id is not None)


def _requested_parent_hierarchy_records(
    records: tuple[SlurmJobRecord, ...],
    *,
    requested: tuple[str, ...],
) -> tuple[SlurmJobRecord, ...]:
    """Retain each exact requested parent plus every scheduler-derived hierarchy row."""
    return tuple(
        record
        for record in records
        if any(
            record.job_id == parent_job_id
            or record.job_id.startswith((f"{parent_job_id}_", f"{parent_job_id}.", f"{parent_job_id}["))
            for parent_job_id in requested
        )
    )


def _requested_json_payload(records: tuple[SlurmJobRecord, ...]) -> dict[str, list[object]]:
    """Persist a requested-ID-scoped JSON payload, excluding scheduler metadata."""
    return {"jobs": [record.raw for record in records]}


def _json_sacct_result(
    result: CommandResult,
    jobs: tuple[SlurmJobRecord, ...],
    *,
    warning: str | None = None,
) -> tuple[SlurmCommandSnapshot, tuple[SlurmJobRecord, ...], tuple[str, ...]]:
    """Return a requested-scoped JSON accounting observation."""
    return (
        SlurmCommandSnapshot(
            kind="sacct",
            argv=result.argv,
            returncode=result.returncode,
            parser="json",
            stderr=result.stderr,
            raw_json=_requested_json_payload(jobs),
            warning=warning,
        ),
        jobs,
        (warning,) if warning is not None else (),
    )


def _has_exact_exit_enrichment(
    sparse_terminal_jobs: tuple[SlurmJobRecord, ...],
    fallback_jobs: tuple[SlurmJobRecord, ...],
    *,
    missing_expected_job_ids: tuple[str, ...] = (),
) -> bool:
    """Require one exact terminal record for every endpoint that triggered fallback."""
    endpoints = tuple(dict.fromkeys((*missing_expected_job_ids, *(job.job_id for job in sparse_terminal_jobs))))
    for endpoint in endpoints:
        matching_exit_records = tuple(
            job
            for job in fallback_jobs
            if job.job_id == endpoint and job.state in TERMINAL_SLURM_STATES and job.exit_code is not None
        )
        if len(matching_exit_records) != 1:
            return False
    return True


def _parent_only_cancelled_absence_closure(
    fallback_jobs: tuple[SlurmJobRecord, ...],
    *,
    missing_array_endpoints: tuple[str, ...],
) -> bool:
    """Recognize exact parent-only cancellation without synthesizing children."""
    if not missing_array_endpoints:
        return False
    affected_parents = tuple(dict.fromkeys(endpoint.split("_", 1)[0] for endpoint in missing_array_endpoints))
    for parent in affected_parents:
        hierarchy = tuple(
            job
            for job in fallback_jobs
            if job.job_id == parent or job.job_id.startswith((f"{parent}_", f"{parent}.", f"{parent}["))
        )
        if len(hierarchy) != 1:
            return False
        row = hierarchy[0]
        if row.job_id != parent or row.state != "CANCELLED" or row.exit_code != "0:0":
            return False
    return True


def _select_job_states(
    requested: tuple[str, ...],
    *,
    squeue_jobs: tuple[SlurmJobRecord, ...],
    sacct_jobs: tuple[SlurmJobRecord, ...],
) -> tuple[SlurmJobState, ...]:
    squeue_states = selected_records_by_job_id(squeue_jobs)
    sacct_states = selected_records_by_job_id(sacct_jobs)
    observed: list[SlurmJobState] = []
    for job_id in requested:
        sacct_state = sacct_states.get(job_id)
        squeue_state = squeue_states.get(job_id)
        selected: SlurmJobRecord | None
        if sacct_state is not None and (is_terminal_slurm_state(sacct_state.state) or squeue_state is None):
            selected = sacct_state
        elif squeue_state is not None:
            selected = squeue_state
        else:
            selected = sacct_state
        if selected is None or selected.state is None:
            continue
        observed.append(
            SlurmJobState(
                job_id=job_id,
                state=normalize_slurm_state(selected.state),
                source=selected.source,
                exit_code=selected.exit_code,
            )
        )
    return tuple(observed)


__all__ = [
    "LEGACY_WIRE_SHAPE_CUTOVER_AT",
    "TERMINAL_SLURM_STATES",
    "CommandResult",
    "CommandRunner",
    "RemoteSlurmTransport",
    "SlurmAction",
    "SlurmActionTransport",
    "SlurmCommandSnapshot",
    "SlurmJobRecord",
    "SlurmJobState",
    "SlurmObservation",
    "SlurmSubmission",
    "SlurmSubmissionRejected",
    "SlurmSubmissionUncertain",
    "artifact_copy_argv",
    "artifact_fetch_argv",
    "command_argv",
    "control_status_from_slurm_state",
    "default_command_runner",
    "is_terminal_slurm_state",
    "legacy_command_argv",
    "normalize_slurm_state",
    "parse_sbatch_job_id",
    "run_probe",
    "shell_argv",
]
