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

"""Contract tests for the public postprocessing example bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_plan import (
    PostprocessingPhasePlan,
    postprocessing_phase_plan_from_mapping,
)
from bspp.orchestration.contract.runplan import load_runplan
from bspp.orchestration.contract.runtime_qualification import (
    runtime_qualification_payload_from_mapping,
)
from bspp.orchestration.control.postprocessing_identity import _logical_inventory

ROOT = Path(__file__).resolve().parents[1]
PHASE_PLAN = ROOT / "skills/examples/run-plans/postprocessing-phase-plan.yaml"

_DOCUMENT_ATTRIBUTES = (
    "legacy_run_plan",
    "acceptance_policy",
    "logical_input_inventory",
    "runtime_qualification",
)

_DOCUMENT_KINDS = {
    "legacy-run-plan",
    "acceptance-policy",
    "logical-input-inventory",
    "runtime-qualification",
}

_AUTOREQUEUE_ACTION_IDS = (
    "postprocessing-01-preflight",
    "postprocessing-02-recipe",
    "postprocessing-03-preprocess",
    "postprocessing-04-slurm",
    "postprocessing-05-analysis-finalize",
    "postprocessing-06-acceptance-tar-payload-parity",
    "postprocessing-07-acceptance-semantic",
    "postprocessing-08-acceptance-verify-evidence",
)


def _load_plan() -> PostprocessingPhasePlan:
    payload = yaml.safe_load(PHASE_PLAN.read_text())
    assert isinstance(payload, dict)
    return postprocessing_phase_plan_from_mapping(payload)


def _companion_bytes(plan: PostprocessingPhasePlan, attribute: str) -> bytes:
    document = getattr(plan, attribute)
    return (PHASE_PLAN.parent / document.path).read_bytes()


def _companion_mapping(plan: PostprocessingPhasePlan, attribute: str) -> dict[str, object]:
    payload = yaml.safe_load(_companion_bytes(plan, attribute))
    assert isinstance(payload, dict)
    return payload


def test_postprocessing_phase_plan_loads() -> None:
    plan = _load_plan()

    assert plan.phase_kind == "postprocessing"
    assert plan.target_cluster == "my-cluster"
    assert {
        plan.legacy_run_plan.document_kind,
        plan.acceptance_policy.document_kind,
        plan.logical_input_inventory.document_kind,
        plan.runtime_qualification.document_kind,
    } == _DOCUMENT_KINDS


@pytest.mark.parametrize("attribute", _DOCUMENT_ATTRIBUTES)
def test_companion_documents_match_pinned_bytes(attribute: str) -> None:
    plan = _load_plan()
    document = getattr(plan, attribute)
    raw = (PHASE_PLAN.parent / document.path).read_bytes()

    assert hashlib.sha256(raw).hexdigest() == document.sha256
    assert len(raw) == document.size_bytes


def test_legacy_run_plan_companion_loads() -> None:
    plan = _load_plan()
    loaded = load_runplan(PHASE_PLAN.parent / plan.legacy_run_plan.path)

    assert loaded.target_cluster == "my-cluster"


def test_acceptance_policy_companion_loads() -> None:
    plan = _load_plan()
    policy = postprocessing_acceptance_policy_from_mapping(_companion_mapping(plan, "acceptance_policy"))

    assert policy.policy_kind == "postprocessing-sealable-v2"


def test_logical_input_inventory_companion_loads() -> None:
    plan = _load_plan()
    inventory = _logical_inventory(_companion_bytes(plan, "logical_input_inventory"))

    assert len(inventory) == 7
    identity, digest, size = inventory["s3-archive-prefix"]
    assert identity == "my-baseline:s3-archive-prefix"
    assert isinstance(digest, str)
    assert digest == "1" * 64
    assert size == 7


def test_runtime_qualification_companion_loads() -> None:
    plan = _load_plan()
    payload = json.loads(_companion_bytes(plan, "runtime_qualification"))
    assert isinstance(payload, dict)

    normalized = runtime_qualification_payload_from_mapping(payload)

    assert normalized["schema_version"] == 1


def test_autorequeue_disabled_default_and_enabled_alternative() -> None:
    plan = _load_plan()

    assert plan.autorequeue_policy.mode == "disabled"
    assert plan.autorequeue_policy.action_ids == ()

    raw_text = PHASE_PLAN.read_text()
    assert "mode: enabled" in raw_text
    for action_id in _AUTOREQUEUE_ACTION_IDS:
        assert action_id in raw_text


def test_postprocessing_example_comment_coverage() -> None:
    """Every field and enum alternative must be enumerated in a comment."""
    text = PHASE_PLAN.read_text()
    comment_lines = [line.split("#", 1)[1] for line in text.splitlines() if "#" in line]
    comment_text = "\n".join(comment_lines)

    expected_tokens = (
        # Phase Plan fields
        "schema_version",
        "phase_kind",
        "target_cluster",
        "output_namespace",
        # LocalAuthorityDocument fields and the four document kinds
        "legacy_run_plan",
        "acceptance_policy",
        "logical_input_inventory",
        "runtime_qualification",
        "document_kind",
        "path",
        "sha256",
        "size_bytes",
        "legacy-run-plan",
        "acceptance-policy",
        "logical-input-inventory",
        "runtime-qualification",
        # autorequeue_policy fields and enum alternatives
        "autorequeue_policy",
        "mode",
        "action_ids",
        "disabled",
        "enabled",
        "postprocessing-09-acceptance-adjudication",
    )
    for token in expected_tokens:
        assert token in comment_text, f"comment coverage missing token: {token}"
