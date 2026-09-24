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

"""Runtime transport-failure adapter and exit-85 authority.

This module is the Runtime side of the classified autorequeue exit.  It owns a
closed typed descriptor for audited transport failures, a pure adapter that maps
only the audited group-A outcomes (timeout, connection reset, temporary DNS, and
HTTP 500/502/503/504) to the Contract's closed observation record, and the sole
exit-85 helper that classifies the observation, writes the create-once
per-restart classification record, and exits with the reserved code 85 for group
A only.

Group-B and non-audited failures are deliberately left to the caller's normal
failure path: the adapter returns ``None`` for them and the exit helper returns
without exiting.  No success evidence is ever produced for a failed incarnation
and ``scontrol`` is never invoked from this module.
"""

from __future__ import annotations

import errno
import re
import sys
from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.postprocessing_failure_classification import (
    PostprocessingTransportFailureKind,
    PostprocessingTransportFailureObservation,
    classify_transport_failure,
)
from bspp.orchestration.runtime.data_movement.common import TransferResult
from bspp.orchestration.runtime.postprocessing.restart_classification import (
    record_restart_classification,
)

TransportFailureKind = Literal[
    "timeout",
    "connection-reset",
    "temporary-dns",
    "http-500",
    "http-502",
    "http-503",
    "http-504",
    "oom",
    "signal",
    "credential-error",
    "unknown",
]

_AUDITED_KIND_MAP: dict[str, PostprocessingTransportFailureKind] = {
    "timeout": "timeout",
    "connection-reset": "connection-reset",
    "temporary-dns": "temporary-dns",
    "http-500": "http-500",
    "http-502": "http-502",
    "http-503": "http-503",
    "http-504": "http-504",
}

_HTTP_STATUS_KIND_MAP: dict[int, TransportFailureKind] = {
    500: "http-500",
    502: "http-502",
    503: "http-503",
    504: "http-504",
}

# The frozen transfer adapters surface an HTTP 500/502/503/504 only as text in
# the captured stdout/stderr tail — never as the POSIX ``returncode`` (which is
# 0..255).  This narrow, word-bounded pattern is the audited extraction for
# those four statuses; it is deliberately NOT a free-form stderr classifier.
_HTTP_STATUS_PATTERN = re.compile(r"\b5(?:00|02|03|04)\b")


class TransportFailure(Exception):  # noqa: N818 - contract name is required by the exit-85 authority
    """A typed, sanitized transport failure that survived bounded retries.

    ``detail`` is a short single-line human label (e.g. ``"connection reset by
    peer"`` or ``"HTTP 503 from s5cmd"``), never free-form stderr bytes.
    """

    def __init__(self, kind: TransportFailureKind, detail: str) -> None:
        if not isinstance(detail, str) or not detail or detail != detail.strip() or "\n" in detail or "\r" in detail:
            raise ValueError("TransportFailure.detail must be a non-empty trimmed single-line string")
        self.kind = kind
        self.detail = detail
        super().__init__(detail)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (type(self), (self.kind, self.detail))


def classify_audited_transport_failure(
    failure: TransportFailure,
) -> PostprocessingTransportFailureObservation | None:
    """Map one audited transport failure to a closed observation, else ``None``.

    Pure and total over the closed descriptor: the seven audited group-A kinds
    map 1:1 to the Contract's observation kinds, while ``oom``, ``signal``,
    ``credential-error``, and ``unknown`` return ``None`` so the caller keeps
    the normal failure path.
    """
    observation_kind = _AUDITED_KIND_MAP.get(failure.kind)
    if observation_kind is None:
        return None
    return PostprocessingTransportFailureObservation(
        failure_kind=observation_kind,
        failure_detail=failure.detail,
    )


def classify_and_exit_for_autorequeue(
    failure: TransportFailure,
    *,
    phase_runspec_path: Path,
    action_id: str,
    command_digest: str,
    task_index: int | None = None,
    restart_count_env: str | None = None,
) -> None:
    """Classify *failure* and, for group A only, record it then ``sys.exit(85)``.

    Group-B and non-audited failures return without exiting (the caller keeps
    the normal failure path).  No success evidence is written and no scheduler
    cancellation is invoked.
    """
    observation = classify_audited_transport_failure(failure)
    if observation is None:
        return None
    classification = classify_transport_failure(observation)
    if classification.group != "A":
        return None
    record_restart_classification(
        phase_runspec_path=phase_runspec_path,
        action_id=action_id,
        command_digest=command_digest,
        classification=classification,
        task_index=task_index,
        restart_count_env=restart_count_env,
    )
    sys.exit(85)


def transport_failure_from_oserror(exc: OSError) -> TransportFailure | None:
    """Map an audited ``OSError`` errno to a typed failure, else ``None``."""
    if exc.errno == errno.ECONNRESET:
        return TransportFailure(kind="connection-reset", detail="connection reset by peer")
    if exc.errno == errno.ETIMEDOUT:
        return TransportFailure(kind="timeout", detail="connection timed out")
    if exc.errno in {errno.EHOSTUNREACH, errno.ENETUNREACH}:
        return TransportFailure(kind="temporary-dns", detail="temporary DNS failure (host/network unreachable)")
    return None


def transport_failure_from_http_status(status: int, *, detail: str) -> TransportFailure | None:
    """Map an audited HTTP status to a typed failure, else ``None``."""
    kind = _HTTP_STATUS_KIND_MAP.get(status)
    if kind is None:
        return None
    return TransportFailure(kind=kind, detail=detail)


def transport_failure_from_result(result: TransferResult) -> TransportFailure | None:
    """Map a failed transfer result to a typed failure using its output text.

    The frozen transfer adapters surface ``returncode`` plus text tails.  A
    POSIX ``returncode`` is 0..255 (or a negative signal), so it can never carry
    the audited HTTP statuses 500/502/503/504; those appear only in the captured
    stdout/stderr tail.  Extract a narrow, word-bounded 500/502/503/504 token
    from those tails; generic non-zero codes stay ``None`` (preserving existing
    behavior).  The OSError-derived path in ``transfer.py`` remains the live
    source for connection-reset/timeout/DNS group-A outcomes.
    """
    if result.ok:
        return None
    text = f"{result.stdout_tail}\n{result.stderr_tail}"
    match = _HTTP_STATUS_PATTERN.search(text)
    if match is None:
        return None
    status = int(match.group(0))
    return transport_failure_from_http_status(
        status,
        detail=f"HTTP {status} from {result.tool}",
    )


def raise_transport_failure_if_audited(result: TransferResult) -> None:
    """Raise :class:`TransportFailure` for an audited result, else no-op."""
    failure = transport_failure_from_result(result)
    if failure is not None:
        raise failure


__all__ = [
    "TransportFailure",
    "TransportFailureKind",
    "classify_and_exit_for_autorequeue",
    "classify_audited_transport_failure",
    "raise_transport_failure_if_audited",
    "transport_failure_from_http_status",
    "transport_failure_from_oserror",
    "transport_failure_from_result",
]
