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

"""Shared types and helpers for data-movement subprocess wrappers."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransferResult:
    """Outcome of a single transfer subprocess invocation."""

    tool: str
    argv: tuple[str, ...]
    returncode: int
    elapsed_s: float
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class PlannedTransfer:
    """The argv of a transfer the caller opted into dry-run mode.

    Returned by the subprocess wrappers when ``dry_run=True`` instead of
    actually executing.
    """

    tool: str
    argv: tuple[str, ...]
    note: str = ""


class ToolMissingError(Exception):
    """Raised when the requested CLI is not on PATH.

    Not a frozen dataclass: Click's ``augment_usage_errors`` may set
    attributes on the exception instance, which a ``@dataclass(frozen=True)``
    would reject with ``FrozenInstanceError``, crashing the error path itself.
    """

    def __init__(self, tool: str, hint: str = "") -> None:
        self.tool = tool
        self.hint = hint
        super().__init__(str(self))

    def __str__(self) -> str:
        base = f"Required CLI '{self.tool}' not found on PATH."
        return f"{base} {self.hint}".rstrip()


def require_tool(tool: str, *, hint: str = "") -> str:
    """Return the resolved path to *tool*, or raise :class:`ToolMissingError`."""
    resolved = shutil.which(tool)
    if resolved is None:
        raise ToolMissingError(tool=tool, hint=hint)
    return resolved


def run_transfer(
    argv: Sequence[str],
    *,
    tool: str,
    env: Mapping[str, str] | None = None,
    tail_bytes: int = 4096,
) -> TransferResult:
    """Run *argv* synchronously and return a :class:`TransferResult`.

    Captures stdout/stderr so the caller can forward error context; only
    the trailing ``tail_bytes`` of each stream are stored to keep the
    result object small.
    """
    started = time.monotonic()
    logger.info("Running %s: %s", tool, " ".join(argv))
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        env=dict(env) if env is not None else None,
        check=False,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        logger.warning(
            "%s exited with code %d in %.1fs; stderr tail: %s",
            tool,
            completed.returncode,
            elapsed,
            completed.stderr[-tail_bytes:].strip(),
        )
    return TransferResult(
        tool=tool,
        argv=tuple(argv),
        returncode=completed.returncode,
        elapsed_s=elapsed,
        stdout_tail=completed.stdout[-tail_bytes:],
        stderr_tail=completed.stderr[-tail_bytes:],
    )


def format_argv(argv: Sequence[str]) -> str:
    """Render *argv* for display (shell-quoted enough for readability)."""
    return " ".join(_shquote(a) for a in argv)


def _shquote(arg: str) -> str:
    if not arg:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@%+=:,./-")
    if all(ch in safe for ch in arg):
        return arg
    return "'" + arg.replace("'", "'\\''") + "'"


def ensure_uri_parent_local(target: Path) -> None:
    """Create *target*'s parent directory if it is a local path."""
    if str(target).startswith(("gs://", "s3://")):
        return
    target.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PlannedTransfer",
    "ToolMissingError",
    "TransferResult",
    "ensure_uri_parent_local",
    "format_argv",
    "require_tool",
    "run_transfer",
]
