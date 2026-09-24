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

"""Compatibility exports for focused postprocessing contract modules."""

from __future__ import annotations

from bspp.orchestration.contract.postprocessing_cancellation_events import (
    PostprocessingCancellationIntendedPayload,
    PostprocessingCancelledPayload,
    PostprocessingJobCancellationRequestIntendedPayload,
    PostprocessingJobCancellationRequestResultPayload,
)
from bspp.orchestration.contract.postprocessing_event import (
    PostprocessingEventPayload,
    PostprocessingEventType,
    PostprocessingPhaseEvent,
    postprocessing_phase_event_from_mapping,
)
from bspp.orchestration.contract.postprocessing_phase_receipt import PostprocessingFinalizedPayload
from bspp.orchestration.contract.postprocessing_retry_events import (
    PostprocessingAttemptRetriedPayload,
    postprocessing_retry_id,
)
from bspp.orchestration.contract.postprocessing_runspec_v2 import PostprocessingPhaseRunSpec
from bspp.orchestration.contract.postprocessing_submission_events import (
    PostprocessingActionDispatchIntendedPayload,
    PostprocessingActionDispatchRejectedPayload,
    PostprocessingActionSubmittedPayload,
    PostprocessingMaterializedPayload,
    PostprocessingSubmissionActionPlan,
    PostprocessingSubmissionIntendedPayload,
)
from bspp.orchestration.contract.postprocessing_terminal_events import (
    PostprocessingActionTerminalObservedPayload,
    PostprocessingArrayParentCancelledObservedPayload,
    PostprocessingTaskTerminalEvidence,
)

__all__ = [
    "PostprocessingActionDispatchIntendedPayload",
    "PostprocessingActionDispatchRejectedPayload",
    "PostprocessingActionSubmittedPayload",
    "PostprocessingActionTerminalObservedPayload",
    "PostprocessingArrayParentCancelledObservedPayload",
    "PostprocessingAttemptRetriedPayload",
    "PostprocessingCancellationIntendedPayload",
    "PostprocessingCancelledPayload",
    "PostprocessingEventPayload",
    "PostprocessingEventType",
    "PostprocessingFinalizedPayload",
    "PostprocessingJobCancellationRequestIntendedPayload",
    "PostprocessingJobCancellationRequestResultPayload",
    "PostprocessingMaterializedPayload",
    "PostprocessingPhaseEvent",
    "PostprocessingPhaseRunSpec",
    "PostprocessingSubmissionActionPlan",
    "PostprocessingSubmissionIntendedPayload",
    "PostprocessingTaskTerminalEvidence",
    "postprocessing_phase_event_from_mapping",
    "postprocessing_retry_id",
]
