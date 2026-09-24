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

"""Autorequeue policy contract and projection tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.postprocessing_autorequeue_policy import (
    POSTPROCESSING_AUTOREQUEUE_DISABLED,
    PostprocessingAutorequeuePolicy,
    postprocessing_autorequeue_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_plan import postprocessing_phase_plan_from_mapping
from bspp.orchestration.control.phase_materialization import materialize_phase
from bspp.orchestration.control.postprocessing_authority_v2 import validate_postprocessing_authority
from bspp.orchestration.control.postprocessing_phase_cancellation import cancel_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_retry import retry_postprocessing_phase
from tests.test_postprocessing_phase_materialization import (
    NOW,
    RUN_ID,
    _document,
    _fixture,
    _materialized_authority,
    _retry_resolution,
)


def test_disabled_policy_requires_empty_allowlist() -> None:
    with pytest.raises(ValueError, match="empty action allowlist"):
        PostprocessingAutorequeuePolicy(mode="disabled", action_ids=("postprocessing-01-preflight",))


def test_enabled_policy_roundtrips_ordered_allowlist() -> None:
    policy = PostprocessingAutorequeuePolicy(
        mode="enabled",
        action_ids=("postprocessing-01-preflight", "postprocessing-04-slurm"),
    )
    assert postprocessing_autorequeue_policy_from_mapping(policy.to_mapping()) == policy
    assert policy.to_mapping()["action_ids"] == ["postprocessing-01-preflight", "postprocessing-04-slurm"]


def test_unknown_action_id_rejected() -> None:
    with pytest.raises(ValueError, match="invalid postprocessing Phase action id"):
        PostprocessingAutorequeuePolicy(mode="enabled", action_ids=("postprocessing-99-bogus",))


def test_duplicate_action_ids_rejected() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        PostprocessingAutorequeuePolicy(
            mode="enabled",
            action_ids=("postprocessing-01-preflight", "postprocessing-01-preflight"),
        )


def test_unsorted_action_ids_rejected() -> None:
    with pytest.raises(ValueError, match="permanent ordinal order"):
        PostprocessingAutorequeuePolicy(
            mode="enabled",
            action_ids=("postprocessing-04-slurm", "postprocessing-01-preflight"),
        )


def test_unknown_field_rejected() -> None:
    payload = POSTPROCESSING_AUTOREQUEUE_DISABLED.to_mapping()
    payload["invented"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        postprocessing_autorequeue_policy_from_mapping(payload)


def test_phase_plan_defaults_missing_policy_to_disabled(tmp_path: Path) -> None:
    plan_path, _profile_path, _source_repo = _fixture(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    assert "autorequeue_policy" not in payload
    plan = postprocessing_phase_plan_from_mapping(payload)
    assert plan.autorequeue_policy.mode == "disabled"
    assert plan.autorequeue_policy.action_ids == ()


def test_materialization_projects_disabled_policy(tmp_path: Path) -> None:
    authority_root = _materialized_authority(tmp_path)
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    assert authority.runspec.payload.autorequeue_policy.mode == "disabled"
    assert authority.runspec.payload.autorequeue_policy.action_ids == ()


def _write_autorequeue_cap(tmp_path: Path) -> None:
    qualification_path = tmp_path / "runtime-qualification.json"
    qualification = json.loads(qualification_path.read_bytes())
    qualification["autorequeue_cap"] = {"requeue_exit": 85, "max_batch_requeue": 5}
    qualification_path.write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n")


def _refresh_runtime_qualification_reference(plan: dict[str, object], tmp_path: Path) -> dict[str, object]:
    plan["runtime_qualification"] = _document("runtime-qualification", tmp_path / "runtime-qualification.json")
    return plan


def test_materialization_projects_enabled_policy_verbatim(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    _write_autorequeue_cap(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    payload = _refresh_runtime_qualification_reference(payload, tmp_path)
    payload["autorequeue_policy"] = {
        "schema_version": 1,
        "mode": "enabled",
        "action_ids": ["postprocessing-01-preflight", "postprocessing-04-slurm"],
    }
    plan_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    authority = validate_postprocessing_authority(authority_root, RUN_ID)
    assert authority.runspec.payload.autorequeue_policy == authority.phase_plan.autorequeue_policy
    assert authority.runspec.payload.autorequeue_policy.mode == "enabled"
    assert authority.runspec.payload.autorequeue_policy.action_ids == (
        "postprocessing-01-preflight",
        "postprocessing-04-slurm",
    )


def test_retry_successor_carries_enabled_policy_verbatim(tmp_path: Path) -> None:
    plan_path, profile_path, source_repo = _fixture(tmp_path)
    _write_autorequeue_cap(tmp_path)
    payload = yaml.safe_load(plan_path.read_text())
    payload = _refresh_runtime_qualification_reference(payload, tmp_path)
    payload["autorequeue_policy"] = {
        "schema_version": 1,
        "mode": "enabled",
        "action_ids": ["postprocessing-01-preflight", "postprocessing-04-slurm"],
    }
    plan_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    authority_root = tmp_path / "authority"
    materialize_phase(
        plan_path,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        phase_run_id_factory=lambda: RUN_ID,
    )
    cancel_postprocessing_phase(RUN_ID, authority_root=authority_root, clock=lambda: NOW)
    predecessor = validate_postprocessing_authority(authority_root, RUN_ID)
    resolver = _retry_resolution(tmp_path=tmp_path, authority=predecessor)
    result = retry_postprocessing_phase(
        RUN_ID,
        authority_root=authority_root,
        config_path=profile_path,
        source_repo=source_repo,
        clock=lambda: NOW,
        runtime_resolver=resolver,
    )
    successor = validate_postprocessing_authority(authority_root, RUN_ID)
    assert successor.attempt_id == "attempt-0002"
    assert result.successor_attempt_id == "attempt-0002"
    assert successor.runspec.payload.autorequeue_policy == predecessor.phase_plan.autorequeue_policy
    assert successor.runspec.payload.autorequeue_policy.mode == "enabled"
