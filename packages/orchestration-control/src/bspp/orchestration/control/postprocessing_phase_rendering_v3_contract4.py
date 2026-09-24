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

"""Renderer contract 4 for portable postprocessing V3 authorities."""

from __future__ import annotations

from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.control.postprocessing_phase_rendering_v3_contract3 import (
    PostprocessingRenderInput,
    _aws_profile_name,
    _render_postprocessing_action_script_with_credential_mounts,
)


def render_postprocessing_action_script_v3_contract4(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> str:
    """Render exclusively from credential locators frozen into V3 authority."""
    if not isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
        raise ValueError("renderer contract 4 requires a postprocessing V3 RunSpec")
    credential_mounts = authority.runspec.credential_mounts
    if credential_mounts is None:
        raise ValueError("renderer contract 4 requires frozen credential mount locators")
    has_aws_mounts = credential_mounts.aws_shared_credentials_file is not None
    if (_aws_profile_name(authority) is not None) != has_aws_mounts:
        raise ValueError("renderer contract 4 credential locators differ from the stored AWS profile reference")
    return _render_postprocessing_action_script_with_credential_mounts(authority, action, credential_mounts)


__all__ = ["render_postprocessing_action_script_v3_contract4"]
