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

"""Renderer contract 5 for postprocessing V3 array-job identity forwarding."""

from __future__ import annotations

from bspp.orchestration.contract.postprocessing_action_contract import PostprocessingRuntimeAction
from bspp.orchestration.contract.postprocessing_runspec_v3 import PostprocessingPhaseRunSpecV3
from bspp.orchestration.control.postprocessing_phase_rendering_v3_contract3 import (
    PostprocessingRenderInput,
    _aws_profile_name,
    _render_postprocessing_action_script_with_credential_mounts,
)


def render_postprocessing_action_script_v3_contract5(
    authority: PostprocessingRenderInput,
    action: PostprocessingRuntimeAction,
) -> str:
    """Render V3 arrays with the closed parent-job-id forwarding contract.

    Scalars deliberately retain environment contract 1, so their renderer-5
    bytes are exactly renderer-4 bytes while submission identity remains
    versioned separately.
    """
    if not isinstance(authority.runspec, PostprocessingPhaseRunSpecV3):
        raise ValueError("renderer contract 5 requires a postprocessing V3 RunSpec")
    credential_mounts = authority.runspec.credential_mounts
    if credential_mounts is None:
        raise ValueError("renderer contract 5 requires frozen credential mount locators")
    has_aws_mounts = credential_mounts.aws_shared_credentials_file is not None
    if (_aws_profile_name(authority) is not None) != has_aws_mounts:
        raise ValueError("renderer contract 5 credential locators differ from the stored AWS profile reference")
    return _render_postprocessing_action_script_with_credential_mounts(
        authority,
        action,
        credential_mounts,
        slurm_environment_contract_version=2 if action.expected_task_indexes else 1,
    )


__all__ = ["render_postprocessing_action_script_v3_contract5"]
