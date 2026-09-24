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

"""Golden baseline tests for generated SLURM render artifacts."""

from __future__ import annotations

import json
import os
import re
from difflib import unified_diff
from pathlib import Path

import pytest

from bspp.orchestration.contract.runspec import load_runspec
from bspp.orchestration.runtime.slurm.finalizer import render_analysis_finalizer
from bspp.orchestration.runtime.slurm.native import render_slurm_native
from tests.test_slurm.runspec_fixtures import _write_finalizer_runspec, _write_runspec

BASELINE_DIR = Path(__file__).parents[1] / "fixtures" / "slurm_render_baselines"
UPDATE_BASELINES = os.environ.get("UPDATE_RENDER_BASELINES") == "1"

RenderOutputs = dict[str, str]


def test_analysis_finalizer_render_matches_normalized_baselines(tmp_path: Path) -> None:
    outputs = _render_analysis_finalizer_outputs(tmp_path)

    _assert_outputs_match_baselines("analysis_finalizer", outputs)


def test_native_worker_render_matches_normalized_baselines(tmp_path: Path) -> None:
    outputs = _render_native_outputs(tmp_path)

    _assert_outputs_match_baselines("native", outputs)


def _render_analysis_finalizer_outputs(tmp_path: Path) -> RenderOutputs:
    spec = load_runspec(_write_finalizer_runspec(tmp_path, high_quality=True))
    script_path = tmp_path / "rendered" / "run_analysis_finalize.sbatch"

    render_analysis_finalizer(spec, dry_run=False, script_path=script_path)

    return {
        "run_analysis_finalize.sbatch": _normalize_script(script_path.read_text(), tmp_path),
        "analysis_finalizer_report.json": _normalize_json(
            (script_path.parent / "analysis_finalizer_report.json").read_text(),
            tmp_path,
        ),
    }


def _render_native_outputs(tmp_path: Path) -> RenderOutputs:
    spec = load_runspec(_write_runspec(tmp_path, phase2_tar=True, s5cmd_numworkers=24))
    script_path = tmp_path / "rendered" / "run_archive.sbatch"

    render_slurm_native(spec, ["archive_a", "archive_b"], dry_run=False, script_path=script_path)

    return {
        "run_archive.sbatch": _normalize_script(script_path.read_text(), tmp_path),
        "slurm_native_report.json": _normalize_json(
            (script_path.parent / "slurm_native_report.json").read_text(),
            tmp_path,
        ),
    }


def _assert_outputs_match_baselines(case_name: str, outputs: RenderOutputs) -> None:
    case_dir = BASELINE_DIR / case_name
    if UPDATE_BASELINES:
        case_dir.mkdir(parents=True, exist_ok=True)
        for filename, actual in outputs.items():
            (case_dir / filename).write_text(actual)
        return

    missing = [filename for filename in outputs if not (case_dir / filename).is_file()]
    if missing:
        pytest.fail(
            "Missing SLURM render baseline(s): "
            + ", ".join(str(case_dir / filename) for filename in missing)
            + ". Re-run with UPDATE_RENDER_BASELINES=1 to create them."
        )

    for filename, actual in outputs.items():
        expected = (case_dir / filename).read_text()
        if actual == expected:
            continue
        diff = "\n".join(
            unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=str(case_dir / filename),
                tofile=f"actual/{case_name}/{filename}",
                lineterm="",
            )
        )
        pytest.fail(f"SLURM render baseline mismatch for {case_name}/{filename}:\n{diff}")


def _normalize_script(payload: str, tmp_path: Path) -> str:
    normalized = _normalize_text(payload, tmp_path)
    normalized = re.sub(r"^(# Source SHA256: ).*$", r"\1<SOURCE_SHA256>", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"^(# Rendered at: ).*$", r"\1<RENDERED_AT>", normalized, flags=re.MULTILINE)
    return _ensure_trailing_newline(normalized)


def _normalize_json(payload: str, tmp_path: Path) -> str:
    normalized = _normalize_value(json.loads(payload), tmp_path)
    return json.dumps(normalized, indent=2, sort_keys=True) + "\n"


def _normalize_value(value: object, tmp_path: Path, *, key: str | None = None) -> object:
    if key == "source_hash":
        return "<SOURCE_SHA256>"
    if isinstance(value, dict):
        return {
            str(child_key): _normalize_value(child_value, tmp_path, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_normalize_value(item, tmp_path) for item in value]
    if isinstance(value, str):
        return _normalize_text(value, tmp_path)
    return value


def _normalize_text(payload: str, tmp_path: Path) -> str:
    return payload.replace(str(tmp_path), "<TMP>")


def _ensure_trailing_newline(payload: str) -> str:
    return payload if payload.endswith("\n") else payload + "\n"
