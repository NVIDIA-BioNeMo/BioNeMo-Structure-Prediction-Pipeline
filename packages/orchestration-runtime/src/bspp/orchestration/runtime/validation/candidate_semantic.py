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

"""Independent candidate semantic comparison for release acceptance."""

from __future__ import annotations

from collections.abc import Mapping

from bspp.orchestration.contract.release_acceptance import SemanticValidationReport


def validate_candidate_semantics(
    *,
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    minimum_pass_count: int,
) -> SemanticValidationReport:
    """Compare canonical semantic values rather than trusting producer success flags."""
    if type(minimum_pass_count) is not int or minimum_pass_count < 0:
        raise ValueError("minimum_pass_count must be a non-negative integer")
    candidate_ids = set(expected)
    unexpected = set(observed) - candidate_ids
    if unexpected:
        raise ValueError(f"unexpected semantic candidates: {sorted(unexpected)}")
    passed = sum(
        1
        for candidate_id in candidate_ids
        if candidate_id in expected and expected[candidate_id] == observed.get(candidate_id)
    )
    return SemanticValidationReport(
        1,
        len(expected),
        minimum_pass_count,
        passed,
        len(expected) - passed,
    )


__all__ = ["validate_candidate_semantics"]
