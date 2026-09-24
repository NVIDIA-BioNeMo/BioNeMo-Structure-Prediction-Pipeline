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

"""Typed read-only projection of postprocessing authority status metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from bspp.orchestration.contract.postprocessing_runspec_v1 import HistoricalPostprocessingPhaseRunSpecV1
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.control.postprocessing_phase_types import ReadablePostprocessingAuthority

PostprocessingContractFamily = Literal[
    "postprocessing-runspec-v1",
    "postprocessing-runspec-v2",
    "postprocessing-runspec-v3",
]


@dataclass(frozen=True)
class PostprocessingAuthorityStatusProjection:
    contract_family: PostprocessingContractFamily
    read_only: bool
    sealed: bool
    current_attempt_projection_complete: bool
    phase_runspec_digest: str
    submission_id: str | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "contract_family": self.contract_family,
            "read_only": self.read_only,
            "sealed": self.sealed,
            "current_attempt_projection_complete": self.current_attempt_projection_complete,
            "phase_runspec_digest": self.phase_runspec_digest,
            "submission_id": self.submission_id,
        }


def project_postprocessing_authority_status(
    authority: ReadablePostprocessingAuthority,
) -> PostprocessingAuthorityStatusProjection:
    historical_v1 = isinstance(authority.runspec, HistoricalPostprocessingPhaseRunSpecV1)
    contract_family: PostprocessingContractFamily = (
        "postprocessing-runspec-v1"
        if historical_v1
        else "postprocessing-runspec-v3"
        if isinstance(authority.runspec, PostprocessingPhaseRunSpecV3)
        else "postprocessing-runspec-v2"
    )
    submission = authority.submission_state
    return PostprocessingAuthorityStatusProjection(
        contract_family=contract_family,
        read_only=historical_v1,
        sealed=authority.sealed,
        current_attempt_projection_complete=authority.current_attempt_projection_complete,
        phase_runspec_digest=authority.runspec.digest,
        submission_id=submission.submission_id if submission else None,
    )


__all__ = [
    "PostprocessingAuthorityStatusProjection",
    "PostprocessingContractFamily",
    "project_postprocessing_authority_status",
]
