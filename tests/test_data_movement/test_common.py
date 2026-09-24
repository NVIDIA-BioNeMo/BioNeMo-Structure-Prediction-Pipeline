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

"""Tests for shared data-movement helpers (common.py)."""

from __future__ import annotations

import dataclasses

import pytest

from bspp.orchestration.runtime.data_movement.common import ToolMissingError

# ---------------------------------------------------------------------------
# #101 regression: ToolMissingError must be Click-augmentation-compatible
# ---------------------------------------------------------------------------


def test_tool_missing_error_is_not_frozen_dataclass() -> None:
    """ToolMissingError must not be a frozen dataclass.

    Click's ``augment_usage_errors`` may set attributes on a caught exception
    instance.  A ``@dataclass(frozen=True)`` raises ``FrozenInstanceError``
    on attribute assignment, which would crash the error path itself.
    """
    assert not dataclasses.is_dataclass(ToolMissingError)


def test_tool_missing_error_allows_attribute_assignment() -> None:
    """Directly verify that setting attributes on a ToolMissingError works.

    This simulates what Click's ``augment_usage_errors`` does: it catches a
    usage error and may attach ``ctx`` or other state to the exception.
    """
    err = ToolMissingError(tool="s5cmd", hint="install s5cmd")
    # Click may set arbitrary attributes; this must not raise.
    err.ctx = "fake-click-context"  # type: ignore[attr-defined]
    assert err.ctx == "fake-click-context"  # type: ignore[attr-defined]


def test_tool_missing_error_preserves_fields_and_message() -> None:
    """Constructor surface and __str__ output must be unchanged."""
    err = ToolMissingError(tool="s5cmd", hint="Install s5cmd (https://github.com/peak/s5cmd).")
    assert err.tool == "s5cmd"
    assert err.hint == "Install s5cmd (https://github.com/peak/s5cmd)."
    assert str(err) == "Required CLI 's5cmd' not found on PATH. Install s5cmd (https://github.com/peak/s5cmd)."


def test_tool_missing_error_default_hint_empty() -> None:
    """With no hint, the message is clean (no trailing space)."""
    err = ToolMissingError(tool="gcloud")
    assert err.tool == "gcloud"
    assert err.hint == ""
    assert str(err) == "Required CLI 'gcloud' not found on PATH."


def test_tool_missing_error_is_exception_subclass() -> None:
    assert issubclass(ToolMissingError, Exception)


def test_tool_missing_error_can_be_raised_and_caught() -> None:
    with pytest.raises(ToolMissingError) as excinfo:
        raise ToolMissingError(tool="dm", hint="internal tool")
    assert excinfo.value.tool == "dm"
    assert "Required CLI 'dm' not found on PATH." in str(excinfo.value)
