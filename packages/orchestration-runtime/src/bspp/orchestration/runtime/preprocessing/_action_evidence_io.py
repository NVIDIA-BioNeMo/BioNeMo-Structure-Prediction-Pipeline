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

"""Strict loading and immutable publication for preprocessing action evidence."""

from __future__ import annotations

import json
from pathlib import Path

from bspp.orchestration.contract.preprocessing_action import (
    PreprocessingChunkActionEvidence,
    preprocessing_chunk_action_evidence_from_mapping,
)

from ._database_placement_errors import DatabasePlacementError
from ._database_placement_evidence_io import (
    _load_canonical_json_at,
    _load_from_path,
    _publish_immutable_exclusive,
)


def load_preprocessing_action_evidence(path: Path) -> PreprocessingChunkActionEvidence:
    """Load one exact descriptor-anchored immutable action-evidence document."""
    try:
        return _load_from_path(
            path,
            description="preprocessing action evidence",
            loader=_load_action_evidence_at,
        )
    except DatabasePlacementError as exc:
        raise ValueError(str(exc)) from exc


def publish_preprocessing_action_evidence(
    evidence: PreprocessingChunkActionEvidence,
    destination: Path,
) -> None:
    """Exclusively publish and descriptor-verify immutable action evidence."""
    try:
        _publish_immutable_exclusive(
            evidence,
            destination,
            serializer=_canonical_action_evidence_bytes,
            loader=_load_action_evidence_at,
            description="preprocessing action evidence",
        )
    except DatabasePlacementError as exc:
        raise ValueError(str(exc)) from exc


def _load_action_evidence_at(
    parent_descriptor: int,
    name: str,
    display_path: Path,
) -> PreprocessingChunkActionEvidence:
    return _load_canonical_json_at(
        parent_descriptor,
        name,
        display_path,
        parser=preprocessing_chunk_action_evidence_from_mapping,
        serializer=_canonical_action_evidence_bytes,
        description="preprocessing action evidence",
    )


def _canonical_action_evidence_bytes(evidence: PreprocessingChunkActionEvidence) -> bytes:
    return (json.dumps(evidence.to_mapping(), indent=2, sort_keys=True) + "\n").encode()


__all__ = ["load_preprocessing_action_evidence", "publish_preprocessing_action_evidence"]
