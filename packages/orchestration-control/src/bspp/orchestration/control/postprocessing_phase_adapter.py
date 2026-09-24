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

"""Public compatibility facade for the postprocessing Phase Lifecycle."""

from __future__ import annotations

from bspp.orchestration.control.postprocessing_authority_v2 import (
    validate_postprocessing_authority,
)
from bspp.orchestration.control.postprocessing_phase_cancellation import cancel_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_materialization import (
    load_postprocessing_phase_plan,
    materialize_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_observation import (
    resume_postprocessing_phase,
    status_postprocessing_phase,
)
from bspp.orchestration.control.postprocessing_phase_rendering import (
    PostprocessingRenderInput,
    postprocessing_action_command_digest,
    render_postprocessing_action_script,
)
from bspp.orchestration.control.postprocessing_phase_submission import submit_postprocessing_phase
from bspp.orchestration.control.postprocessing_phase_types import (
    PostprocessingAuthority,
    PostprocessingLifecycleResult,
    PostprocessingPhaseMaterializationResult,
)

__all__ = [
    "PostprocessingAuthority",
    "PostprocessingLifecycleResult",
    "PostprocessingPhaseMaterializationResult",
    "PostprocessingRenderInput",
    "cancel_postprocessing_phase",
    "load_postprocessing_phase_plan",
    "materialize_postprocessing_phase",
    "postprocessing_action_command_digest",
    "render_postprocessing_action_script",
    "resume_postprocessing_phase",
    "status_postprocessing_phase",
    "submit_postprocessing_phase",
    "validate_postprocessing_authority",
]
