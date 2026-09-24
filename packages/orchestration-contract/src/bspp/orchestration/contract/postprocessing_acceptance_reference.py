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

"""Digest-bound reference to one postprocessing acceptance policy snapshot."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bspp.orchestration.contract._postprocessing_validation import _schema
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION

_SHA256 = re.compile(r"[0-9a-f]{64}")
_POLICY_ID = re.compile(r"postprocessing-acceptance-policy-[0-9a-f]{64}")


@dataclass(frozen=True)
class PostprocessingAcceptanceSnapshotReference:
    location: str
    sha256: str
    size_bytes: int
    semantic_digest: str
    policy_id: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if not self.location.startswith("attempts/") or not self.location.endswith("/acceptance-policy.json"):
            raise ValueError("acceptance snapshot location must be attempt-relative")
        for value in (self.sha256, self.semantic_digest):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("acceptance snapshot digests must be lowercase SHA-256")
        if (
            _POLICY_ID.fullmatch(self.policy_id) is None
            or self.policy_id.removeprefix("postprocessing-acceptance-policy-") != self.semantic_digest
        ):
            raise ValueError("acceptance snapshot policy id must bind its semantic digest")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            raise ValueError("acceptance snapshot size must be positive")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "location": self.location,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "semantic_digest": self.semantic_digest,
            "policy_id": self.policy_id,
        }


__all__ = ["PostprocessingAcceptanceSnapshotReference"]
