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

"""Keep new Phase-lifecycle tests on reviewed public or external seams."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

_TESTS = Path(__file__).parent
_SCOPED_NAMES = {
    "test_postprocessing_autorequeue_integration.py",
    "test_postprocessing_evidence_transfer.py",
    "test_postprocessing_finalization_bundle_contract.py",
    "test_postprocessing_finalization_bundle_runtime.py",
    "test_postprocessing_phase_acceptance.py",
    "test_postprocessing_phase_contract.py",
    "test_postprocessing_phase_materialization.py",
    "test_postprocessing_restarts.py",
    "test_postprocessing_scheduler_evidence.py",
}
_EXPECTED_EXTERNAL_PATCHES = Counter(
    {
        ("RemoteSlurmTransport", "query_observation_best_effort"): 13,
        ("RemoteSlurmTransport", "request_job_cancellation"): 6,
        ("RemoteSlurmTransport", "stage_immutable_artifact"): 4,
        ("RemoteSlurmTransport", "command"): 3,
        ("RemoteSlurmTransport", "submit_action"): 3,
        ("RemoteSlurmTransport", "query_submissions_by_correlation"): 2,
        ("Path", "home"): 4,
        ("phase_artifacts.subprocess", "run"): 1,
        ("shutil", "copy2"): 1,
    }
)


def test_postprocessing_phase_tests_patch_only_reviewed_transport_boundaries() -> None:
    observed: Counter[tuple[str, str]] = Counter()
    forbidden: list[str] = []
    for name in sorted(_SCOPED_NAMES):
        path = _TESTS / name
        for node in ast.walk(ast.parse(path.read_text(), filename=name)):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in {"patch", "setattr", "delattr"}:
                forbidden.append(f"{name}:{node.lineno}: {node.func.id}")
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "object"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "patch"
            ):
                forbidden.append(f"{name}:{node.lineno}: patch.object")
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr not in {
                "setattr",
                "delattr",
                "setitem",
                "delitem",
                "patch",
                "patch.object",
            }:
                continue
            if node.func.attr != "setattr" or len(node.args) < 2:
                forbidden.append(f"{name}:{node.lineno}: {ast.unparse(node.func)}")
                continue
            owner = ast.unparse(node.args[0])
            attribute_node = node.args[1]
            if not isinstance(attribute_node, ast.Constant) or not isinstance(attribute_node.value, str):
                forbidden.append(f"{name}:{node.lineno}: dynamic setattr")
                continue
            observed[(owner, attribute_node.value)] += 1

    assert forbidden == []
    assert observed == _EXPECTED_EXTERNAL_PATCHES
