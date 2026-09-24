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

"""Single source of permanent postprocessing Phase action identifiers."""

from __future__ import annotations

import re

POSTPROCESSING_STEP_ORDINALS: dict[str, int] = {
    "preflight": 1,
    "recipe": 2,
    "preprocess": 3,
    "slurm": 4,
    "analysis-finalize": 5,
    "acceptance-tar-payload-parity": 6,
    "acceptance-semantic": 7,
    "acceptance-verify-evidence": 8,
    "acceptance-adjudication": 9,
}
POSTPROCESSING_ACTION_IDS: dict[str, str] = {
    name: f"postprocessing-{ordinal:02d}-{name}" for name, ordinal in POSTPROCESSING_STEP_ORDINALS.items()
}
_ACTION_ID = re.compile(r"postprocessing-([0-9]{2})-([a-z0-9-]+)")


def postprocessing_action_id(step_name: str) -> str:
    try:
        return POSTPROCESSING_ACTION_IDS[step_name]
    except KeyError as exc:
        raise ValueError(f"unsupported postprocessing phase step: {step_name!r}") from exc


def validate_postprocessing_action_id(action_id: str) -> str:
    match = _ACTION_ID.fullmatch(action_id)
    if match is None or POSTPROCESSING_ACTION_IDS.get(match.group(2)) != action_id:
        raise ValueError(f"invalid postprocessing Phase action id: {action_id!r}")
    return action_id


__all__ = [
    "POSTPROCESSING_ACTION_IDS",
    "POSTPROCESSING_STEP_ORDINALS",
    "postprocessing_action_id",
    "validate_postprocessing_action_id",
]
