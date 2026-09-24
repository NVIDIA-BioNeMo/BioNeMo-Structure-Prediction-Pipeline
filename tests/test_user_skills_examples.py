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

"""Contract tests for the public user-skills example run plans."""

from __future__ import annotations

from pathlib import Path

import yaml

from bspp.orchestration.contract.phase import phase_plan_from_mapping

ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_PLAN = ROOT / "skills" / "examples" / "run-plans" / "preprocessing-phase-plan.yaml"


def test_preprocessing_example_loads() -> None:
    """The preprocessing example must load through the strict PhasePlan loader."""
    raw = yaml.safe_load(PREPROCESSING_PLAN.read_text())
    assert isinstance(raw, dict)

    plan = phase_plan_from_mapping(raw)

    assert plan.phase_kind == "preprocessing"
    assert plan.target_cluster == "my-cluster"
    assert plan.input_location.path == plan.payload.work_plan.input.source_path


def test_preprocessing_example_comment_coverage() -> None:
    """Every field and enum alternative must be enumerated in a comment."""
    text = PREPROCESSING_PLAN.read_text()
    comment_lines = [line.split("#", 1)[1] for line in text.splitlines() if "#" in line]
    comment_text = "\n".join(comment_lines)

    expected_tokens = (
        "requested_tranches",
        "records_per_chunk",
        "nodes",
        "gpus_per_node",
        "strict-two-line",
        "normalize-multiline",
        "stage-required",
        "stage-preferred",
        "direct",
        "publish-to-s3",
        "local",
        "verified-local-file",
        "verified-remote-file",
        "s3_publish_prefix",
    )
    for token in expected_tokens:
        assert token in comment_text, f"comment coverage missing token: {token}"
