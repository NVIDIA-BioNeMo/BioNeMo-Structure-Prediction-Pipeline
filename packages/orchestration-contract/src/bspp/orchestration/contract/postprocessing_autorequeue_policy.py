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

"""Closed typed autorequeue policy for postprocessing Phase Attempts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from bspp.orchestration.contract._postprocessing_validation import (
    _fields,
    _schema,
    _str,
    _str_list,
)
from bspp.orchestration.contract.postprocessing_phase_ids import (
    POSTPROCESSING_ACTION_IDS,
    validate_postprocessing_action_id,
)
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PostprocessingAutorequeueMode = Literal["disabled", "enabled"]


@dataclass(frozen=True)
class PostprocessingAutorequeuePolicy:
    """Attempt-bound autorequeue policy: a mode plus a canonical action allowlist."""

    mode: PostprocessingAutorequeueMode
    action_ids: tuple[str, ...]
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version, type(self).__name__)
        if self.mode not in ("disabled", "enabled"):
            raise ValueError(f"postprocessing autorequeue mode must be 'disabled' or 'enabled': {self.mode!r}")
        if not isinstance(self.action_ids, tuple):
            raise ValueError("postprocessing autorequeue action_ids must be an immutable tuple")
        for action_id in self.action_ids:
            validate_postprocessing_action_id(action_id)
        if POSTPROCESSING_ACTION_IDS["acceptance-adjudication"] in self.action_ids:
            raise ValueError("postprocessing autorequeue policy may only describe Actions 01--08")
        if len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("postprocessing autorequeue action_ids must be unique")
        ordered = tuple(sorted(self.action_ids, key=tuple(POSTPROCESSING_ACTION_IDS.values()).index))
        if ordered != self.action_ids:
            raise ValueError("postprocessing autorequeue action_ids must be in permanent ordinal order")
        if self.mode == "disabled" and self.action_ids != ():
            raise ValueError("postprocessing autorequeue disabled policy requires an empty action allowlist")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "action_ids": list(self.action_ids),
        }


POSTPROCESSING_AUTOREQUEUE_DISABLED = PostprocessingAutorequeuePolicy(mode="disabled", action_ids=())


def postprocessing_autorequeue_policy_from_mapping(payload: Mapping[str, object]) -> PostprocessingAutorequeuePolicy:
    _fields(payload, {"schema_version", "mode", "action_ids"}, "PostprocessingAutorequeuePolicy")
    return PostprocessingAutorequeuePolicy(
        mode=cast("PostprocessingAutorequeueMode", _str(payload, "mode")),
        action_ids=_str_list(payload, "action_ids"),
        schema_version=validate_schema_version(
            payload.get("schema_version"), record_name="PostprocessingAutorequeuePolicy"
        ),
    )


__all__ = [
    "POSTPROCESSING_AUTOREQUEUE_DISABLED",
    "PostprocessingAutorequeueMode",
    "PostprocessingAutorequeuePolicy",
    "postprocessing_autorequeue_policy_from_mapping",
]
