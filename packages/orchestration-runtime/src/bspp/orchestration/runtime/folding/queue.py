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

"""Public non-executing folding queue planner.

Port Baseline:
``3864d0eda67e70979b8e48f00ed6a08f9e71c59e:folding/openfold-pipeline/scripts/preprocess_trt_bionemo.py:356-400``.
"""

from __future__ import annotations

from collections.abc import Iterable

from bspp.orchestration.contract.folding_index import FoldingIndexRecord
from bspp.orchestration.contract.folding_queue import FoldingQueueConfig, FoldingQueuePlan
from bspp.orchestration.runtime.folding.queue_planning import (
    plan_exact_length_assignments,
    plan_round_robin_assignments,
    plan_runtime_balanced_assignments,
)


def plan_folding_queue(records: Iterable[FoldingIndexRecord], config: FoldingQueueConfig) -> FoldingQueuePlan:
    """Plan one of the three pinned strategies without running a worker."""
    if config.strategy == "exact_length":
        return plan_exact_length_assignments(records, config)
    if config.strategy == "runtime_balanced":
        return plan_runtime_balanced_assignments(records, config)
    return plan_round_robin_assignments(records, config)


__all__ = ["plan_folding_queue"]
