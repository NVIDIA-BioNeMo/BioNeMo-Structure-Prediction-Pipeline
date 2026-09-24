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

"""Versioned deterministic renderers for postprocessing V2 authorities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.postprocessing_runspec import ReadablePostprocessingPhaseRunSpec
from bspp.orchestration.contract.postprocessing_runspec_v2 import PostprocessingPhaseRunSpec
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.contract.postprocessing_submission_events import (
    POSTPROCESSING_V2_RENDERER_CONTRACT_SUPPORTED,
    POSTPROCESSING_V3_RENDERER_CONTRACT_SUPPORTED,
)
from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.control.postprocessing_identity import resolve_action_semantic
from bspp.orchestration.control.postprocessing_phase_rendering_v2_contract1 import _cluster_attempt_root


@dataclass(frozen=True)
class PostprocessingRenderInput:
    """The immutable V2 authority inputs needed by a renderer contract."""

    runspec: ReadablePostprocessingPhaseRunSpec
    legacy_runspec: RunSpec

    @property
    def phase_run_id(self) -> str:
        return self.runspec.phase_run_id

    @property
    def attempt_id(self) -> str:
        return self.runspec.attempt_id


PostprocessingRenderer = Callable[[PostprocessingRenderInput, PostprocessingRuntimeAction], str]


def render_postprocessing_action_script(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
    *,
    renderer_contract_version: int = min(POSTPROCESSING_V2_RENDERER_CONTRACT_SUPPORTED),
) -> str:
    """Render V2 with its historical default or V3 with its explicit contract."""
    if isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
        if renderer_contract_version not in POSTPROCESSING_V3_RENDERER_CONTRACT_SUPPORTED:
            raise ValueError("postprocessing V3 supports only renderer contracts 3, 4, and 5")
        return postprocessing_v3_renderer(renderer_contract_version)(authority, action)
    if not isinstance(authority.runspec, PostprocessingPhaseRunSpec):
        raise ValueError("historical postprocessing V1 authority requires its frozen V1 renderer")
    if renderer_contract_version not in POSTPROCESSING_V2_RENDERER_CONTRACT_SUPPORTED:
        raise ValueError("postprocessing V2 supports only renderer contracts 1 and 2")
    stored_semantic = next(
        (item for item in authority.runspec.payload.action_semantics.actions if item.action_id == action.action_id),
        None,
    )
    resolved_semantic = resolve_action_semantic(
        action,
        legacy_runspec=authority.legacy_runspec,
        normalized_arguments=authority.runspec.payload.action_semantics.semantic_fields,
    )
    if stored_semantic != resolved_semantic:
        raise ValueError(f"stored action semantics differ from rendered command for {action.action_id!r}")
    return postprocessing_v2_renderer(renderer_contract_version)(authority, action)


def _renderer_v1(authority: PostprocessingRenderInput, action: PostprocessingRuntimeAction) -> str:
    from bspp.orchestration.control.postprocessing_phase_rendering_v2_contract1 import (
        render_postprocessing_action_script_v2_contract1,
    )

    return render_postprocessing_action_script_v2_contract1(cast(Any, authority), action)


def _renderer_v2(authority: PostprocessingRenderInput, action: PostprocessingRuntimeAction) -> str:
    from bspp.orchestration.control.postprocessing_phase_rendering_v2_contract2 import (
        render_postprocessing_action_script_v2_contract2,
    )

    return render_postprocessing_action_script_v2_contract2(cast(Any, authority), action)


def _renderer_v3(authority: PostprocessingRenderInput, action: PostprocessingRuntimeAction) -> str:
    from bspp.orchestration.control.postprocessing_phase_rendering_v3_contract3 import (
        render_postprocessing_action_script_v3_contract3,
    )

    return render_postprocessing_action_script_v3_contract3(cast(Any, authority), action)


def _renderer_v4(authority: PostprocessingRenderInput, action: PostprocessingRuntimeAction) -> str:
    from bspp.orchestration.control.postprocessing_phase_rendering_v3_contract4 import (
        render_postprocessing_action_script_v3_contract4,
    )

    return render_postprocessing_action_script_v3_contract4(cast(Any, authority), action)


def _renderer_v5(authority: PostprocessingRenderInput, action: PostprocessingRuntimeAction) -> str:
    from bspp.orchestration.control.postprocessing_phase_rendering_v3_contract5 import (
        render_postprocessing_action_script_v3_contract5,
    )

    return render_postprocessing_action_script_v3_contract5(cast(Any, authority), action)


def postprocessing_action_command_digest(
    authority: PostprocessingRenderInput, action: PostprocessingRuntimeAction
) -> str:
    """Return the stable action-command digest shared by renderer contracts."""
    from bspp.orchestration.control.postprocessing_phase_rendering_v2_contract1 import (
        postprocessing_action_command_digest as contract1_digest,
    )

    return contract1_digest(cast(Any, authority), action)


POSTPROCESSING_V2_RENDERERS: Mapping[int, PostprocessingRenderer] = MappingProxyType(
    {
        1: _renderer_v1,
        2: _renderer_v2,
    }
)
if frozenset(POSTPROCESSING_V2_RENDERERS) != POSTPROCESSING_V2_RENDERER_CONTRACT_SUPPORTED:
    raise RuntimeError("postprocessing V2 renderer registry differs from the supported contract versions")


def postprocessing_v2_renderer(version: int) -> PostprocessingRenderer:
    """Return the exact V2 renderer implementation for a frozen version."""
    try:
        return POSTPROCESSING_V2_RENDERERS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported postprocessing V2 renderer contract version: {version}") from exc


POSTPROCESSING_V3_RENDERERS: Mapping[int, PostprocessingRenderer] = MappingProxyType(
    {
        3: _renderer_v3,
        4: _renderer_v4,
        5: _renderer_v5,
    }
)
if frozenset(POSTPROCESSING_V3_RENDERERS) != POSTPROCESSING_V3_RENDERER_CONTRACT_SUPPORTED:
    raise RuntimeError("postprocessing V3 renderer registry differs from the supported contract versions")


def postprocessing_v3_renderer(version: int) -> PostprocessingRenderer:
    """Return one exact V3 renderer implementation."""
    try:
        return POSTPROCESSING_V3_RENDERERS[version]
    except KeyError as exc:
        raise ValueError(f"unsupported postprocessing V3 renderer contract version: {version}") from exc


__all__ = [
    "POSTPROCESSING_V2_RENDERERS",
    "POSTPROCESSING_V3_RENDERERS",
    "PostprocessingRenderInput",
    "_cluster_attempt_root",
    "postprocessing_action_command_digest",
    "postprocessing_v2_renderer",
    "postprocessing_v3_renderer",
    "render_postprocessing_action_script",
]
