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

"""Unit tests for the data-movement backend discovery seam."""

from __future__ import annotations

import importlib.metadata
import inspect

import pytest

from bspp.orchestration.runtime.data_movement import backends
from bspp.orchestration.runtime.data_movement.gcs import transfer as gcs_transfer
from bspp.orchestration.runtime.data_movement.s3 import transfer as s3_transfer


def test_entry_point_group_is_nonempty_string() -> None:
    assert isinstance(backends.BACKEND_ENTRY_POINT_GROUP, str)
    assert backends.BACKEND_ENTRY_POINT_GROUP


def test_backend_unavailable_error_subclasses_runtime_error() -> None:
    assert issubclass(backends.BackendUnavailableError, RuntimeError)


def test_backend_unavailable_error_carries_requested_name() -> None:
    error = backends.BackendUnavailableError("dm")
    assert error.name == "dm"
    assert "dm" in str(error)


def test_backend_unavailable_error_retains_hint_once() -> None:
    hint = "select --tool s5cmd or --tool gcloud instead"
    error = backends.BackendUnavailableError("dm", hint=hint)

    assert error.name == "dm"
    assert error.hint == hint
    message = str(error)
    assert message.count("data-movement backend") == 1
    assert message.count(hint) == 1
    assert message.startswith("data-movement backend 'dm' is not available")
    # The hint is never folded back into ``name``.
    assert error.name == "dm"
    assert hint not in error.name


def test_get_backend_resolves_builtin_s5cmd() -> None:
    assert backends.get_backend("s5cmd") is s3_transfer


def test_get_backend_resolves_builtin_gcloud() -> None:
    assert backends.get_backend("gcloud") is gcs_transfer


def test_get_backend_dm_absent_raises_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: importlib.metadata.EntryPoints([]),
    )

    with pytest.raises(backends.BackendUnavailableError) as excinfo:
        backends.get_backend("dm")

    assert excinfo.value.name == "dm"
    assert excinfo.value.hint is None
    assert "dm" in str(excinfo.value)
    assert not isinstance(excinfo.value, ImportError)


def test_get_backend_resolves_registered_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_point = importlib.metadata.EntryPoint(
        name="fake",
        value="bspp.orchestration.runtime.data_movement.gcs.transfer",
        group=backends.BACKEND_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: importlib.metadata.EntryPoints([entry_point]),
    )

    assert backends.get_backend("fake") is gcs_transfer


def test_get_backend_wraps_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_point = importlib.metadata.EntryPoint(
        name="broken",
        value="bspp.orchestration.runtime.data_movement.does_not_exist",
        group=backends.BACKEND_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: importlib.metadata.EntryPoints([entry_point]),
    )

    with pytest.raises(backends.BackendUnavailableError) as excinfo:
        backends.get_backend("broken")

    assert "broken" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_get_backend_wraps_non_import_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_point = importlib.metadata.EntryPoint(
        name="broken-attr",
        value="os:does_not_exist",
        group=backends.BACKEND_ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: importlib.metadata.EntryPoints([entry_point]),
    )

    with pytest.raises(backends.BackendUnavailableError) as excinfo:
        backends.get_backend("broken-attr")

    assert "broken-attr" in str(excinfo.value)
    assert not isinstance(excinfo.value.__cause__, ImportError)


def test_upload_and_track_has_no_direct_dm_import() -> None:
    from bspp.orchestration.runtime.postprocessing import upload_and_track

    source = inspect.getsource(upload_and_track)
    assert "data_movement import dm" not in source
    assert "data_movement.dm" not in source


def test_cli_upload_tool_defaults_are_public() -> None:
    from bspp.orchestration.runtime import cli as cli_mod

    source = inspect.getsource(cli_mod)
    assert 'default="dm"' not in source
    assert 'default="s5cmd"' in source
    assert 'default="gcloud"' in source


def test_python_api_defaults_match_first_public_tool_by_stage() -> None:
    """Python API defaults follow DATA_PLACEMENT_TOOLS_BY_STAGE; dm stays explicit."""
    from bspp.orchestration.contract.data_placement import DATA_PLACEMENT_TOOLS_BY_STAGE
    from bspp.orchestration.runtime.postprocessing import gcs_preflight, upload_and_track
    from bspp.orchestration.runtime.postprocessing.runspec_reports import plan_fallback_upload

    defaults = (
        ("s3", upload_and_track.upload_s3),
        ("gcs", upload_and_track.upload_gcs),
        ("gcs", gcs_preflight.build_gcs_preflight_report),
        ("s3", plan_fallback_upload),
    )
    for stage, func in defaults:
        default = inspect.signature(func).parameters["tool"].default
        assert default == DATA_PLACEMENT_TOOLS_BY_STAGE[stage][0], (
            f"{func.__name__} default {default!r} must be the first public tool for stage {stage!r}"
        )
        assert default != "dm"

    # dm remains an explicitly selectable historical/internal label.
    assert "dm" in DATA_PLACEMENT_TOOLS_BY_STAGE["s3"]
    assert "dm" in DATA_PLACEMENT_TOOLS_BY_STAGE["gcs"]
