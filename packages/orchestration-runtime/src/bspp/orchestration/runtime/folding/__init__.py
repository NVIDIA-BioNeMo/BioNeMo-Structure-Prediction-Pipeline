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

"""Folding Compatibility Port package anchor.

Port Baseline: 3864d0eda67e70979b8e48f00ed6a08f9e71c59e. This package owns local
folding data behavior. The anchor imports no upstream source and does not
execute the OpenFold or TRT-BioNeMo Scientific Kernels.
"""

from __future__ import annotations

from bspp.orchestration.runtime.folding.a3m import ParsedA3m, parse_a3m
from bspp.orchestration.runtime.folding.archive import (
    parse_archive_manifest_jsonl,
    plan_folding_archives,
    render_archive_manifest_jsonl,
)
from bspp.orchestration.runtime.folding.batch_info import normalize_batch_info
from bspp.orchestration.runtime.folding.checkpoint import (
    CURRENT_COMPLETION_FIELDS,
    CURRENT_FAILURE_FIELDS,
    LEGACY_COMPLETION_FIELDS,
    LEGACY_FAILURE_FIELDS,
    load_checkpoint_state,
    merge_checkpoint_records,
    normalize_checkpoint_protein_id,
    write_merged_checkpoint_views,
)
from bspp.orchestration.runtime.folding.indexing import index_folding_a3ms
from bspp.orchestration.runtime.folding.lifecycle import (
    interpret_folding_status_lifecycle,
    load_lifecycle_checkpoint_state,
    plan_folding_archive_lifecycle,
    plan_folding_index_lifecycle,
    plan_folding_merge_lifecycle,
    plan_folding_resume_lifecycle,
    plan_folding_submit_lifecycle,
    preflight_folding_lifecycle,
)
from bspp.orchestration.runtime.folding.queue import plan_folding_queue
from bspp.orchestration.runtime.folding.queue_planning import (
    plan_exact_length_assignments,
    plan_round_robin_assignments,
    plan_runtime_balanced_assignments,
)
from bspp.orchestration.runtime.folding.queue_render import (
    render_folding_queue_manifest,
    render_folding_queues,
)
from bspp.orchestration.runtime.folding.results import normalize_result_protein_id, scan_folding_results
from bspp.orchestration.runtime.folding.resume import classify_failure, plan_checkpoint_resume

__all__ = [
    "CURRENT_COMPLETION_FIELDS",
    "CURRENT_FAILURE_FIELDS",
    "LEGACY_COMPLETION_FIELDS",
    "LEGACY_FAILURE_FIELDS",
    "ParsedA3m",
    "classify_failure",
    "index_folding_a3ms",
    "interpret_folding_status_lifecycle",
    "load_checkpoint_state",
    "load_lifecycle_checkpoint_state",
    "merge_checkpoint_records",
    "normalize_batch_info",
    "normalize_checkpoint_protein_id",
    "normalize_result_protein_id",
    "parse_a3m",
    "parse_archive_manifest_jsonl",
    "plan_checkpoint_resume",
    "plan_exact_length_assignments",
    "plan_folding_archive_lifecycle",
    "plan_folding_archives",
    "plan_folding_index_lifecycle",
    "plan_folding_merge_lifecycle",
    "plan_folding_queue",
    "plan_folding_resume_lifecycle",
    "plan_folding_submit_lifecycle",
    "plan_round_robin_assignments",
    "plan_runtime_balanced_assignments",
    "preflight_folding_lifecycle",
    "render_archive_manifest_jsonl",
    "render_folding_queue_manifest",
    "render_folding_queues",
    "scan_folding_results",
    "write_merged_checkpoint_views",
]
