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

"""Create-once per-restart postprocessing classification evidence.

Each failed Runtime incarnation of a group-A transport failure writes exactly
one immutable classification record keyed by action/task plus a validated
restart ordinal.  The ordinal arrives from the rendered per-action restart-count
file (written by renderer contract 3 from the outer batch shell's
``SLURM_RESTART_COUNT``, which cannot cross the closed ``env -i`` allowlist as an
inherited variable) and is bounded by the qualified ``max_batch_requeue`` cap.
Replay is idempotent; any differing pre-existing record is a collision.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from bspp.orchestration.contract.postprocessing_action_contract import (
    PostprocessingRuntimeAction,
)
from bspp.orchestration.contract.postprocessing_autorequeue_contract import (
    POSTPROCESSING_RESTART_EVIDENCE_PREFIX,
    postprocessing_restart_classification_filename,
)
from bspp.orchestration.contract.postprocessing_failure_classification import (
    PostprocessingFailureClassification,
)
from bspp.orchestration.contract.postprocessing_phase_ids import POSTPROCESSING_ACTION_IDS
from bspp.orchestration.contract.postprocessing_restart_classification import (
    PostprocessingRestartClassificationEvidence,
    postprocessing_restart_classification_evidence_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec import (
    ExecutablePostprocessingPhaseRunSpec,
)
from bspp.orchestration.runtime.postprocessing.finalization_io import (
    _absolute_directory,
    _canonical_bytes,
    _canonical_mapping,
    _load_runspec,
    _stable_file_bytes,
    _timestamp,
    _write_create_once,
)
from bspp.orchestration.runtime.postprocessing.runtime_action_evidence import (
    _action,
    _validate_task_index,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ACTION09_ID = POSTPROCESSING_ACTION_IDS["acceptance-adjudication"]


def record_restart_classification(
    *,
    phase_runspec_path: Path,
    action_id: str,
    command_digest: str,
    classification: PostprocessingFailureClassification,
    task_index: int | None = None,
    restart_count_env: str | None = None,
) -> PostprocessingRestartClassificationEvidence:
    """Create once one exact per-restart classification record for a failed action."""
    runspec = _load_runspec(phase_runspec_path)
    action = _action(runspec, action_id)
    if action.action_id == _ACTION09_ID:
        raise ValueError("Action 09 must only use the non-success prepublication assembly witness")
    _validate_task_index(action, task_index)
    _sha(command_digest, "postprocessing restart classification command digest")
    restart_ordinal = _restart_ordinal(restart_count_env, runspec)
    evidence_root = _absolute_directory(Path(runspec.payload.attempt_paths.evidence_dir), "evidence root", create=True)
    path = _restart_classification_evidence_path(evidence_root, action, task_index, restart_ordinal)
    if path.exists() or os.path.lexists(path):
        existing = postprocessing_restart_classification_evidence_from_mapping(
            _canonical_mapping(_stable_file_bytes(path), label=f"restart classification evidence {path}")
        )
        _reconcile_restart_classification_evidence(
            existing,
            runspec=runspec,
            action=action,
            command_digest=command_digest,
            classification=classification,
            task_index=task_index,
            restart_ordinal=restart_ordinal,
        )
        return existing
    evidence = PostprocessingRestartClassificationEvidence(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_runspec_digest=runspec.digest,
        action_graph_digest=runspec.payload.action_graph_digest,
        action_id=action.action_id,
        runtime_action_digest=action.digest,
        task_index=task_index,
        restart_ordinal=restart_ordinal,
        classification=classification,
        classified_at=_timestamp(),
    )
    _write_create_once(path, _canonical_bytes(evidence.to_mapping()))
    return evidence


def _restart_ordinal(
    restart_count_env: str | None,
    runspec: ExecutablePostprocessingPhaseRunSpec,
) -> int:
    if restart_count_env is None:
        ordinal = 0
    else:
        if not isinstance(restart_count_env, str) or re.fullmatch(r"[0-9]+", restart_count_env) is None:
            raise ValueError("postprocessing restart count must be a non-negative integer string")
        ordinal = int(restart_count_env)
    cap = runspec.payload.qualified_runtime.max_batch_requeue
    if cap is None:
        raise ValueError("postprocessing restart classification requires a qualified max_batch_requeue cap")
    if ordinal > cap:
        raise ValueError("postprocessing restart ordinal exceeds the qualified max_batch_requeue cap")
    return ordinal


def _restart_classification_evidence_path(
    evidence_root: Path,
    action: PostprocessingRuntimeAction,
    task_index: int | None,
    restart_ordinal: int,
) -> Path:
    return (
        evidence_root
        / POSTPROCESSING_RESTART_EVIDENCE_PREFIX
        / action.action_id
        / postprocessing_restart_classification_filename(task_index, restart_ordinal)
    )


def _reconcile_restart_classification_evidence(
    evidence: PostprocessingRestartClassificationEvidence,
    *,
    runspec: ExecutablePostprocessingPhaseRunSpec,
    action: PostprocessingRuntimeAction,
    command_digest: str,
    classification: PostprocessingFailureClassification,
    task_index: int | None,
    restart_ordinal: int,
) -> None:
    # command_digest is validated by the caller but intentionally not stored or
    # compared here: command identity is already bound by the full RunSpec digest
    # and the per-action runtime digest.
    if (
        evidence.phase_run_id != runspec.phase_run_id
        or evidence.attempt_id != runspec.attempt_id
        or evidence.phase_runspec_digest != runspec.digest
        or evidence.action_graph_digest != runspec.payload.action_graph_digest
        or evidence.action_id != action.action_id
        or evidence.runtime_action_digest != action.digest
        or evidence.task_index != task_index
        or evidence.restart_ordinal != restart_ordinal
        or evidence.classification != classification
    ):
        raise ValueError("postprocessing restart classification evidence differs from the frozen action/task identity")


def _sha(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")


__all__ = ["record_restart_classification"]
