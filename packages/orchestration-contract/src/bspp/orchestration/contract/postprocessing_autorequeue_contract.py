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

"""Cross-plane autorequeue handshake contract.

Control renders these names and paths into the action script; Runtime reads them
back at the exit-85 boundary.  Both planes must import from here so a rename
cannot silently disarm autorequeue.
"""

from __future__ import annotations

AUTOREQUEUE_PHASE_RUNSPEC_ENV = "BSPP_AUTOREQUEUE_PHASE_RUNSPEC"
AUTOREQUEUE_ACTION_ID_ENV = "BSPP_AUTOREQUEUE_ACTION_ID"
AUTOREQUEUE_COMMAND_DIGEST_ENV = "BSPP_AUTOREQUEUE_COMMAND_DIGEST"
AUTOREQUEUE_RESTART_COUNT_FILE_ENV = "BSPP_AUTOREQUEUE_RESTART_COUNT_FILE"
AUTOREQUEUE_TASK_INDEX_ENV = "BSPP_AUTOREQUEUE_TASK_INDEX"

AUTOREQUEUE_ENV_NAMES: tuple[str, ...] = (
    AUTOREQUEUE_PHASE_RUNSPEC_ENV,
    AUTOREQUEUE_ACTION_ID_ENV,
    AUTOREQUEUE_COMMAND_DIGEST_ENV,
    AUTOREQUEUE_RESTART_COUNT_FILE_ENV,
    AUTOREQUEUE_TASK_INDEX_ENV,
)

POSTPROCESSING_RESTART_EVIDENCE_PREFIX = "phase-actions/restarts"
POSTPROCESSING_RESTART_COUNT_FILENAME = "restart-count"


def postprocessing_restart_evidence_relative_dir(action_id: str) -> str:
    return f"{POSTPROCESSING_RESTART_EVIDENCE_PREFIX}/{action_id}"


def postprocessing_restart_classification_filename(task_index: int | None, restart_ordinal: int) -> str:
    if task_index is None:
        return f"restart-{restart_ordinal:010d}-classification.json"
    return f"restart-{task_index:010d}-{restart_ordinal:010d}-classification.json"


__all__ = [
    "AUTOREQUEUE_ACTION_ID_ENV",
    "AUTOREQUEUE_COMMAND_DIGEST_ENV",
    "AUTOREQUEUE_ENV_NAMES",
    "AUTOREQUEUE_PHASE_RUNSPEC_ENV",
    "AUTOREQUEUE_RESTART_COUNT_FILE_ENV",
    "AUTOREQUEUE_TASK_INDEX_ENV",
    "POSTPROCESSING_RESTART_COUNT_FILENAME",
    "POSTPROCESSING_RESTART_EVIDENCE_PREFIX",
    "postprocessing_restart_classification_filename",
    "postprocessing_restart_evidence_relative_dir",
]
