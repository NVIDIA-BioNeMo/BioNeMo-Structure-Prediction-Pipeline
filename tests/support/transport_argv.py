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

"""Test helpers for the base64-wrapped SSH login-shell transport form.

The SSH transport wraps the remote command as
``bash -lc 'eval "$(echo <base64> | base64 -d)"'`` so non-POSIX login shells
cannot mangle it. These helpers build and unwrap that exact wire form so tests
can pin it without duplicating the private implementation.
"""

from __future__ import annotations

import base64
import shlex

_WRAPPED_PREFIX = "bash -lc 'eval \"$(echo "
_WRAPPED_SUFFIX = " | base64 -d)\"'"


def wrap_remote_command(command: str) -> str:
    """Return the exact transport payload for COMMAND (structure pinned literally)."""
    encoded = base64.b64encode(command.encode()).decode()
    return _WRAPPED_PREFIX + encoded + _WRAPPED_SUFFIX


def unwrap_remote_command(payload: str) -> str:
    """Decode a wrapped transport payload back to the inner command.

    Raises ValueError when the payload is not in the wrapped form.
    """
    if not (payload.startswith(_WRAPPED_PREFIX) and payload.endswith(_WRAPPED_SUFFIX)):
        raise ValueError(f"not a wrapped transport payload: {payload!r}")
    encoded = payload[len(_WRAPPED_PREFIX) : -len(_WRAPPED_SUFFIX)]
    return base64.b64decode(encoded.encode()).decode()


def maybe_unwrap_remote_command(payload: str) -> str:
    """Return the inner command when PAYLOAD is wrapped, else PAYLOAD unchanged.

    For test fakes that see both wrapped ssh login-shell commands and plain
    scp-style argv tails.
    """
    try:
        return unwrap_remote_command(payload)
    except ValueError:
        return payload


def legacy_remote_command(command: str) -> str:
    """Return the legacy transport payload for COMMAND (structure pinned literally).

    Historically, the SSH transport wrapped the remote command as
    ``bash -lc 'COMMAND'`` (with shlex quoting).  This helper exists only for
    tests that pin legacy replay tolerance and must never be used to build new
    invocations.
    """
    return "bash -lc " + shlex.quote(command)
