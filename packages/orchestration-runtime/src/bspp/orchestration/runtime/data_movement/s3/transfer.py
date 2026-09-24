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

"""S3 transfers via the ``s5cmd`` CLI (against an S3 endpoint).

s5cmd handles concurrency and multipart natively; we just build the
right argv and let it run. Credentials come from explicit environment variables
or AWS shared credentials files (see
:mod:`bspp.orchestration.runtime.data_movement.s3.client`).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from bspp.orchestration.runtime.data_movement.common import (
    PlannedTransfer,
    TransferResult,
    ensure_uri_parent_local,
    require_tool,
    run_transfer,
)
from bspp.orchestration.runtime.data_movement.s3.client import (
    S3_ENDPOINT_ENV,
    MissingS3CredentialsError,
    S3Credentials,
    load_credentials_from_env,
)

_TOOL_HINT = "Install s5cmd (https://github.com/peak/s5cmd) or use the container image."


def _build_env(credentials: S3Credentials | None, extra: Mapping[str, str] | None) -> dict[str, str]:
    env = dict(os.environ)
    if credentials is None:
        credentials = load_credentials_from_env()
    env.update(credentials.as_env())
    if extra:
        env.update(extra)
    return env


def _dry_run_argv(
    s5cmd_path: str,
    credentials: S3Credentials | None,
    numworkers: int | None,
) -> list[str]:
    """Build argv for dry-run without requiring the tool or credentials."""
    argv: list[str] = [s5cmd_path, "--endpoint-url"]
    if credentials is not None:
        argv.append(credentials.endpoint_url)
    else:
        argv.append("<credentials not available>")
    if numworkers is not None and numworkers > 0:
        argv.extend(["--numworkers", str(numworkers)])
    return argv


def _base_argv(s5cmd_path: str, credentials: S3Credentials, numworkers: int | None) -> list[str]:
    argv: list[str] = [s5cmd_path, "--endpoint-url", credentials.endpoint_url]
    if numworkers is not None and numworkers > 0:
        argv.extend(["--numworkers", str(numworkers)])
    return argv


def _try_load_credentials() -> S3Credentials | None:
    """Best-effort credential load for dry-run; returns ``None`` if unavailable."""
    try:
        return load_credentials_from_env()
    except MissingS3CredentialsError:
        return None


def cp(
    src: str | Path,
    dst: str | Path,
    *,
    credentials: S3Credentials | None = None,
    numworkers: int | None = None,
    extra_args: Iterable[str] = (),
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> TransferResult | PlannedTransfer:
    """Copy *src* to *dst* via ``s5cmd cp``.

    Either endpoint may be an ``s3://`` URI or a local path. Globs in
    *src* (e.g. ``path/*``) are supported by s5cmd natively.
    """
    if dry_run:
        s5cmd_path = shutil.which("s5cmd") or "s5cmd"
        creds = credentials or _try_load_credentials()
        argv = _dry_run_argv(s5cmd_path, creds, numworkers)
        argv.append("cp")
        argv.extend(extra_args)
        argv.extend([str(src), str(dst)])
        return PlannedTransfer(
            tool="s5cmd",
            argv=tuple(argv),
            note=f"cp (endpoint via {S3_ENDPOINT_ENV})",
        )

    s5cmd = require_tool("s5cmd", hint=_TOOL_HINT)
    creds = credentials or load_credentials_from_env()
    argv = _base_argv(s5cmd, creds, numworkers)
    argv.append("cp")
    argv.extend(extra_args)
    argv.extend([str(src), str(dst)])

    if not str(dst).startswith("s3://"):
        ensure_uri_parent_local(Path(str(dst)))

    return run_transfer(argv, tool="s5cmd", env=_build_env(creds, env))


def sync(
    src: str | Path,
    dst: str | Path,
    *,
    credentials: S3Credentials | None = None,
    numworkers: int | None = None,
    extra_args: Iterable[str] = (),
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> TransferResult | PlannedTransfer:
    """Sync *src* to *dst* via ``s5cmd sync`` (directory-level)."""
    if dry_run:
        s5cmd_path = shutil.which("s5cmd") or "s5cmd"
        creds = credentials or _try_load_credentials()
        argv = _dry_run_argv(s5cmd_path, creds, numworkers)
        argv.append("sync")
        argv.extend(extra_args)
        argv.extend([str(src), str(dst)])
        return PlannedTransfer(tool="s5cmd", argv=tuple(argv), note="sync")

    s5cmd = require_tool("s5cmd", hint=_TOOL_HINT)
    creds = credentials or load_credentials_from_env()
    argv = _base_argv(s5cmd, creds, numworkers)
    argv.append("sync")
    argv.extend(extra_args)
    argv.extend([str(src), str(dst)])

    return run_transfer(argv, tool="s5cmd", env=_build_env(creds, env))


def run_command_file(
    command_file: Path,
    *,
    credentials: S3Credentials | None = None,
    numworkers: int | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> TransferResult | PlannedTransfer:
    """Invoke ``s5cmd run <command_file>`` for batched operations.

    The file must contain one s5cmd sub-command per line (e.g.
    ``cp /local/path s3://bucket/key``). This is the form emitted by the
    legacy ``upload_and_track.py`` for the s5cmd upload path.
    """
    if dry_run:
        s5cmd_path = shutil.which("s5cmd") or "s5cmd"
        creds = credentials or _try_load_credentials()
        argv = _dry_run_argv(s5cmd_path, creds, numworkers)
        argv.extend(["run", str(command_file)])
        return PlannedTransfer(
            tool="s5cmd",
            argv=tuple(argv),
            note=f"run {command_file.name}",
        )

    s5cmd = require_tool("s5cmd", hint=_TOOL_HINT)
    creds = credentials or load_credentials_from_env()
    argv = _base_argv(s5cmd, creds, numworkers)
    argv.extend(["run", str(command_file)])

    return run_transfer(argv, tool="s5cmd", env=_build_env(creds, env))


def write_cp_command_file(
    pairs: Sequence[tuple[str, str]],
    output_path: Path,
) -> Path:
    """Write an s5cmd command file with one ``cp src dst`` line per pair."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"cp {src} {dst}" for src, dst in pairs]
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


__all__ = ["cp", "run_command_file", "sync", "write_cp_command_file"]
