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

"""Canonical scheduler identities for postprocessing Runtime Actions."""

from __future__ import annotations

from pathlib import PurePosixPath

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.postprocessing_runspec import ReadablePostprocessingPhaseRunSpec


def postprocessing_scheduler_correlation_token(
    runspec: ReadablePostprocessingPhaseRunSpec,
    action: PostprocessingRuntimeAction,
    *,
    legacy: bool = False,
) -> str:
    """Return the stable scheduler token bound to one Attempt and action.

    ``legacy=True`` reproduces the pre-rename ``bspp-pp-`` prefix so
    historical V1/V2 submission events replay byte-identically.
    """
    digest = canonical_mapping_digest(
        {
            "schema_version": 1,
            "phase_run_id": runspec.phase_run_id,
            "attempt_id": runspec.attempt_id,
            "phase_runspec_digest": runspec.digest,
            "action_id": action.action_id,
        }
    )
    prefix = "afcdb-pp-" if legacy else "bspp-pp-"
    return f"{prefix}{digest[:48]}"


def postprocessing_cluster_action_script(
    runspec: ReadablePostprocessingPhaseRunSpec,
    action: PostprocessingRuntimeAction,
    *,
    legacy: bool = False,
) -> PurePosixPath:
    """Return the immutable cluster-side script location for one action.

    ``legacy=True`` reproduces the pre-rename ``afcdb-phase-runs`` directory so
    historical V1/V2 submission events replay byte-identically.
    """
    phase_runs_dir = "afcdb-phase-runs" if legacy else "bspp-phase-runs"
    attempt_root = (
        PurePosixPath(runspec.cluster.staging_root) / phase_runs_dir / runspec.phase_run_id / runspec.attempt_id
    )
    return attempt_root / "actions" / f"{action.action_id}.sbatch"


__all__ = ["postprocessing_cluster_action_script", "postprocessing_scheduler_correlation_token"]
