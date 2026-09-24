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

"""Independent verification entry point for governed execution evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bspp.orchestration.contract.submission_evidence import (
    EvidenceIndex,
    EvidenceIndexValidation,
    build_evidence_index,
    stable_read_evidence_file,
)


def verify_execution_provenance_tree(root: Path, index: EvidenceIndex) -> EvidenceIndexValidation:
    """Verify the exact indexed tree without trusting worker status flags."""
    try:
        actual = build_evidence_index(root)
    except (OSError, ValueError) as exc:
        return EvidenceIndexValidation(False, (str(exc),))
    expected = {entry.path: entry.sha256 for entry in index.entries}
    observed = {entry.path: entry.sha256 for entry in actual.entries if entry.path != "evidence-index.json"}
    issues = [f"missing evidence path: {path}" for path in sorted(set(expected) - set(observed))]
    issues.extend(f"unexpected evidence path: {path}" for path in sorted(set(observed) - set(expected)))
    issues.extend(
        f"evidence digest mismatch: {path}"
        for path in sorted(set(expected) & set(observed))
        if expected[path] != observed[path]
    )
    return EvidenceIndexValidation(not issues, tuple(issues))


def verify_processing_provenance_boundary(
    root: Path,
    index: EvidenceIndex,
    *,
    processing_step_indices: tuple[int, ...],
) -> EvidenceIndexValidation:
    """Verify frozen processing evidence while permitting later acceptance attempts."""
    try:
        actual = build_evidence_index(root)
        selected: dict[str, str] = {}
        for entry in actual.entries:
            parts = Path(entry.path).parts
            if len(parts) >= 3 and parts[0] == "submissions":
                try:
                    step_index = int(parts[1].split("-", 1)[0])
                except ValueError as exc:
                    raise ValueError(f"invalid attempt directory in evidence: {parts[1]}") from exc
                if step_index in processing_step_indices:
                    selected[entry.path] = entry.sha256
            elif len(parts) == 2 and parts[0] == "submission-coordinator" and parts[1].endswith(".json"):
                data = stable_read_evidence_file(root, Path(entry.path), max_bytes=64 * 1024)
                if hashlib.sha256(data).hexdigest() != entry.sha256:
                    raise ValueError(f"unstable coordinator evidence: {entry.path}")
                payload = json.loads(data)
                token = payload.get("token") if isinstance(payload, dict) else None
                if not isinstance(token, dict) or type(token.get("step_index")) is not int:
                    raise ValueError(f"invalid coordinator evidence: {entry.path}")
                if token["step_index"] in processing_step_indices:
                    selected[entry.path] = entry.sha256
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return EvidenceIndexValidation(False, (str(exc),))
    expected = {entry.path: entry.sha256 for entry in index.entries}
    issues = [f"missing processing evidence path: {path}" for path in sorted(set(expected) - set(selected))]
    issues.extend(f"unexpected processing evidence path: {path}" for path in sorted(set(selected) - set(expected)))
    issues.extend(
        f"processing evidence digest mismatch: {path}"
        for path in sorted(set(expected) & set(selected))
        if expected[path] != selected[path]
    )
    return EvidenceIndexValidation(not issues, tuple(issues))


__all__ = ["verify_execution_provenance_tree", "verify_processing_provenance_boundary"]
