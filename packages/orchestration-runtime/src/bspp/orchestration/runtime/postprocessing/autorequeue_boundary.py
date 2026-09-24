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

"""Runtime boundary that closes the exit-85 loop from rendered action context.

The governed ``env -i`` allowlist deliberately omits ``SLURM_RESTART_COUNT``,
so the restart ordinal cannot reach Runtime as an inherited environment
variable.  Renderer contract 3 instead writes the value into a per-action
restart-count file from the outer batch shell (where the real Slurm
environment still exists) and exports a closed context for the Runtime
boundary.  This module reads that context, fails closed when it is absent or
malformed, and otherwise classifies a group-A transport failure and exits 85.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.postprocessing_autorequeue_contract import (
    AUTOREQUEUE_ACTION_ID_ENV,
    AUTOREQUEUE_COMMAND_DIGEST_ENV,
    AUTOREQUEUE_PHASE_RUNSPEC_ENV,
    AUTOREQUEUE_RESTART_COUNT_FILE_ENV,
    AUTOREQUEUE_TASK_INDEX_ENV,
)
from bspp.orchestration.runtime.postprocessing.failure_adapter import (
    TransportFailure,
    classify_and_exit_for_autorequeue,
)

_PHASE_RUNSPEC_ENV = AUTOREQUEUE_PHASE_RUNSPEC_ENV
_ACTION_ID_ENV = AUTOREQUEUE_ACTION_ID_ENV
_COMMAND_DIGEST_ENV = AUTOREQUEUE_COMMAND_DIGEST_ENV
_RESTART_COUNT_FILE_ENV = AUTOREQUEUE_RESTART_COUNT_FILE_ENV
_TASK_INDEX_ENV = AUTOREQUEUE_TASK_INDEX_ENV

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGITS = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class _AutorequeueContext:
    phase_runspec: str
    action_id: str
    command_digest: str
    restart_count_file: str
    task_index: int | None


def handle_autorequeue_transport_failure(failure: TransportFailure) -> None:
    """Apply the autorequeue boundary for one caught TransportFailure.

    Fail closed: an absent or partial rendered context, or an unreadable or
    malformed restart-count file, leaves the normal failure path untouched (no
    exit 85).  Otherwise delegate to the shared classifier, which exits 85 for
    group A only.
    """
    context = _read_context(os.environ)
    if context is None:
        return None
    restart_count = _read_restart_count(context.restart_count_file)
    if restart_count is None:
        return None
    classify_and_exit_for_autorequeue(
        failure,
        phase_runspec_path=Path(context.phase_runspec),
        action_id=context.action_id,
        command_digest=context.command_digest,
        task_index=context.task_index,
        restart_count_env=restart_count,
    )


def _read_context(environ: Mapping[str, str]) -> _AutorequeueContext | None:
    phase_runspec = environ.get(_PHASE_RUNSPEC_ENV)
    action_id = environ.get(_ACTION_ID_ENV)
    command_digest = environ.get(_COMMAND_DIGEST_ENV)
    restart_count_file = environ.get(_RESTART_COUNT_FILE_ENV)
    if phase_runspec is None or action_id is None or command_digest is None or restart_count_file is None:
        return None
    if not phase_runspec or not action_id or not restart_count_file:
        return None
    if _SHA256.fullmatch(command_digest) is None:
        return None
    task_index: int | None = None
    raw_task_index = environ.get(_TASK_INDEX_ENV)
    if raw_task_index:
        if _DIGITS.fullmatch(raw_task_index) is None:
            return None
        task_index = int(raw_task_index)
    return _AutorequeueContext(
        phase_runspec=phase_runspec,
        action_id=action_id,
        command_digest=command_digest,
        restart_count_file=restart_count_file,
        task_index=task_index,
    )


def _read_restart_count(path: str) -> str | None:
    try:
        text = Path(path).read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        return None
    if _DIGITS.fullmatch(text) is None:
        return None
    return text


__all__ = ["handle_autorequeue_transport_failure"]
