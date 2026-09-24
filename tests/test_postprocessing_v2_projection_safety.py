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

"""Fail-closed classification and execution gates for persisted V2 authority."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import bspp.orchestration.control.postprocessing_phase_retry as retry_module
import bspp.orchestration.control.postprocessing_phase_submission as submission_module
from bspp.orchestration.control.postprocessing_v2_projection_safety import (
    classify_v2_projection,
    require_safe_v2_projection,
)
from tests.test_postprocessing_phase_materialization import _fixture

PHASE_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"


def _run_plan(path: Path, *, selector: str | None = "task853") -> bytes:
    _phase_plan, _profile, _source = _fixture(path.parent)
    mapping = yaml.safe_load(path.read_bytes())
    mapping["dataset"]["name"] = mapping["dataset"]["run_id"] if selector is None else selector
    document = yaml.safe_dump(mapping, sort_keys=False).encode()
    path.write_bytes(document)
    return document


def _authority(
    path: Path,
    document: bytes,
    *,
    projected_selector: str = "task853",
    projection_complete: bool = True,
) -> object:
    reference = SimpleNamespace(
        path=str(path),
        size_bytes=len(document),
        sha256=hashlib.sha256(document).hexdigest(),
    )
    return SimpleNamespace(
        phase_plan=SimpleNamespace(legacy_run_plan=reference),
        phase_run_id=PHASE_RUN_ID,
        legacy_runspec=SimpleNamespace(dataset=SimpleNamespace(name=projected_selector)),
        runspec=object(),
        current_attempt_projection_complete=projection_complete,
        sealed=False,
        status="materialized",
    )


def test_v2_projection_classification_is_proof_based_and_fail_closed(tmp_path: Path) -> None:
    plan_path = tmp_path / "run-plan.yaml"
    document = _run_plan(plan_path)

    assert classify_v2_projection(_authority(plan_path, document)).status == "safe"
    assert (
        classify_v2_projection(_authority(plan_path, document, projected_selector="attempt-scoped-broken-name")).status
        == "broken"
    )
    assert classify_v2_projection(_authority(tmp_path / "missing.yaml", document)).status == "unverifiable"
    tampered = _authority(plan_path, document)
    plan_path.write_bytes(document + b"# changed\n")
    assert classify_v2_projection(tampered).status == "unverifiable"


def test_v2_projection_gate_reports_rematerialization_instruction(tmp_path: Path) -> None:
    plan_path = tmp_path / "run-plan.yaml"
    document = _run_plan(plan_path)
    broken = _authority(plan_path, document, projected_selector="attempt-scoped-broken-name")

    with pytest.raises(ValueError, match="rematerialize the Phase Plan as an explicit V3 authority"):
        require_safe_v2_projection(broken)


def test_v2_projection_allows_legitimate_equal_authored_name_and_run_id(tmp_path: Path) -> None:
    plan_path = tmp_path / "run-plan.yaml"
    document = _run_plan(plan_path, selector=None)
    selector = yaml.safe_load(document)["dataset"]["run_id"]

    assert classify_v2_projection(_authority(plan_path, document, projected_selector=selector)).status == "safe"


def test_first_submission_and_retry_gate_v2_before_any_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "run-plan.yaml"
    document = _run_plan(plan_path)
    broken = _authority(plan_path, document, projected_selector="attempt-scoped-broken-name")

    monkeypatch.setattr(submission_module, "reject_historical_postprocessing_mutation", lambda *_args: None)
    monkeypatch.setattr(submission_module, "postprocessing_operation_lock", lambda *_args: nullcontext())
    monkeypatch.setattr(submission_module, "_validate_postprocessing_authority", lambda *_args: broken)
    monkeypatch.setattr(submission_module, "_submission_view", lambda *_args: None)
    with pytest.raises(ValueError, match="V2 projection is broken"):
        submission_module.submit_postprocessing_phase(PHASE_RUN_ID, authority_root=tmp_path)

    monkeypatch.setattr(retry_module, "reject_historical_postprocessing_mutation", lambda *_args: None)
    monkeypatch.setattr(retry_module, "_postprocessing_operation_lock", lambda *_args: nullcontext())
    monkeypatch.setattr(retry_module, "require_postprocessing_v2_authority", lambda *_args: broken)
    with pytest.raises(ValueError, match="V2 projection is broken"):
        retry_module.retry_postprocessing_phase(
            PHASE_RUN_ID,
            authority_root=tmp_path,
            config_path=tmp_path / "profiles.yaml",
        )


def test_v2_retry_projection_recovery_does_not_reopen_first_effect_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "run-plan.yaml"
    document = _run_plan(plan_path)
    incomplete = _authority(
        plan_path,
        document,
        projected_selector="attempt-scoped-broken-name",
        projection_complete=False,
    )
    completed = SimpleNamespace(marker="completed")
    expected = SimpleNamespace(status="materialized")

    monkeypatch.setattr(retry_module, "reject_historical_postprocessing_mutation", lambda *_args: None)
    monkeypatch.setattr(retry_module, "_postprocessing_operation_lock", lambda *_args: nullcontext())
    monkeypatch.setattr(retry_module, "require_postprocessing_v2_authority", lambda *_args: incomplete)
    monkeypatch.setattr(retry_module, "_publish_retry_projection", lambda *_args, **_kwargs: completed)
    monkeypatch.setattr(retry_module, "_retry_result", lambda authority: expected if authority is completed else None)

    def unexpected_gate(_authority: object) -> None:
        raise AssertionError("durable Retry projection recovery must not reopen the V2 first-effect gate")

    monkeypatch.setattr(retry_module, "require_safe_v2_projection", unexpected_gate)

    assert (
        retry_module.retry_postprocessing_phase(
            PHASE_RUN_ID,
            authority_root=tmp_path,
            config_path=tmp_path / "profiles.yaml",
        )
        is expected
    )
