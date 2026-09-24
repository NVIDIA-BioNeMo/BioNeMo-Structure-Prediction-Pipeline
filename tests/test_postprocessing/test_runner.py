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

"""Tests for bspp.orchestration.runtime.postprocessing.runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from bspp.orchestration.runtime.postprocessing.runner import (
    check_meta_json_for_null,
    prefilter_batch,
    run_pipeline,
)

# --- check_meta_json_for_null ---


def test_check_meta_json_for_null_clean(tmp_path: Path) -> None:
    """Valid meta JSON with no nulls returns False."""
    meta = tmp_path / "AF-001-meta_v1.json"
    data = {
        "plddt": [80.5, 90.1, 75.3],
        "pae": [[1.0, 2.0], [2.0, 1.0]],
        "max_pae": 5.0,
        "ptm": 0.8,
        "iptm": 0.7,
    }
    meta.write_text(json.dumps(data))
    assert check_meta_json_for_null(meta) is False


def test_check_meta_json_for_null_with_null(tmp_path: Path) -> None:
    """Meta JSON with null plddt value returns True."""
    meta = tmp_path / "AF-001-meta_v1.json"
    data = {
        "plddt": [80.5, None, 75.3],
        "pae": [[1.0, 2.0], [2.0, 1.0]],
        "max_pae": 5.0,
    }
    meta.write_text(json.dumps(data))
    assert check_meta_json_for_null(meta) is True


def test_check_meta_json_for_null_with_null_pae(tmp_path: Path) -> None:
    """Meta JSON with null in PAE array returns True."""
    meta = tmp_path / "AF-001-meta_v1.json"
    data = {
        "plddt": [80.5, 90.1],
        "pae": [[1.0, None], [2.0, 1.0]],
        "max_pae": 5.0,
    }
    meta.write_text(json.dumps(data))
    assert check_meta_json_for_null(meta) is True


def test_check_meta_json_for_null_missing_file(tmp_path: Path) -> None:
    """Non-existent file returns False."""
    meta = tmp_path / "nonexistent.json"
    assert check_meta_json_for_null(meta) is False


def test_check_meta_json_for_null_only_null(tmp_path: Path) -> None:
    """Array containing only null returns True."""
    meta = tmp_path / "AF-001-meta_v1.json"
    data = {"plddt": [None], "pae": [[1.0]], "max_pae": 5.0}
    meta.write_text(json.dumps(data))
    assert check_meta_json_for_null(meta) is True


# --- prefilter_batch ---


def test_prefilter_batch(tmp_path: Path) -> None:
    """Mix of clean and null models correctly splits into good and bad."""
    # Clean model
    clean_meta = tmp_path / "AF-001-meta_v1.json"
    clean_meta.write_text(json.dumps({"plddt": [80.5], "pae": [[1.0]], "max_pae": 5.0}))

    # Model with null
    bad_meta = tmp_path / "AF-002-meta_v1.json"
    bad_meta.write_text(json.dumps({"plddt": [None], "pae": [[1.0]], "max_pae": 5.0}))

    # Model with no meta file (passes: non-existent -> not null)
    # AF-003 has no meta file

    good, bad = prefilter_batch(["AF-001", "AF-002", "AF-003"], tmp_path)
    assert good == ["AF-001", "AF-003"]
    assert len(bad) == 1
    assert bad[0][0] == "AF-002"
    assert "null" in bad[0][1]


def test_prefilter_batch_all_clean(tmp_path: Path) -> None:
    """All clean models return in good_ids, no failures."""
    for mid in ["AF-001", "AF-002", "AF-003"]:
        meta = tmp_path / f"{mid}-meta_v1.json"
        meta.write_text(json.dumps({"plddt": [80.5, 90.1], "pae": [[1.0, 2.0]], "max_pae": 5.0}))

    good, bad = prefilter_batch(["AF-001", "AF-002", "AF-003"], tmp_path)
    assert good == ["AF-001", "AF-002", "AF-003"]
    assert bad == []


# --- run_pipeline ---


def test_run_pipeline_subprocess_not_found(tmp_path: Path) -> None:
    """When production_pipeline.py is not found, return exit code 127."""
    with patch("bspp.orchestration.runtime.postprocessing.runner.Path.exists", return_value=False):
        # Mock afdb_toolkit to return a path where the script doesn't exist
        exit_code = run_pipeline(
            tmp_path / "input",
            tmp_path / "output",
            tmp_path / "manifest.csv",
            workers=1,
        )
    assert exit_code == 127


def test_run_pipeline_requires_explicit_workers(tmp_path: Path) -> None:
    """Callers must choose worker count instead of inheriting a silent default."""
    with pytest.raises(TypeError, match="workers"):
        run_pipeline(  # type: ignore[call-arg]
            tmp_path / "input",
            tmp_path / "output",
            tmp_path / "manifest.csv",
        )
