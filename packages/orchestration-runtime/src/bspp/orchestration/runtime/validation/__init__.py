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

"""Validation helpers for post-processed outputs."""

from __future__ import annotations

from bspp.orchestration.runtime.validation.count_outputs import (
    CountReport,
    ShardCount,
    count_shard_outputs,
    failed_shard_ids,
)
from bspp.orchestration.runtime.validation.coverage import (
    CoverageReport,
    ShardCoverage,
    validate_coverage,
)
from bspp.orchestration.runtime.validation.timing import (
    StageTiming,
    TimingSummary,
    aggregate,
    load_shard_results,
)

__all__ = [
    "CountReport",
    "CoverageReport",
    "ShardCount",
    "ShardCoverage",
    "StageTiming",
    "TimingSummary",
    "aggregate",
    "count_shard_outputs",
    "failed_shard_ids",
    "load_shard_results",
    "validate_coverage",
]
