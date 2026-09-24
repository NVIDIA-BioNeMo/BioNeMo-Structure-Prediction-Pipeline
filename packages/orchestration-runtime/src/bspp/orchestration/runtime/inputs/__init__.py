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

"""Input preparation helpers for RunSpec-driven BSPP runs."""

from __future__ import annotations

from bspp.orchestration.runtime.inputs.archives import (
    ArchiveCoverage,
    ArchiveStagingItem,
    ArchiveStagingPlan,
    archives_for_dataset,
    archives_for_runspec,
    archives_for_runspec_staging,
    archives_from_staging_dir,
    plan_archive_staging,
    select_archives_for_array,
)
from bspp.orchestration.runtime.inputs.references import (
    DownloadPlan,
    ReferenceArtifact,
    ReferenceStatus,
    check_references,
    ensure_references,
    plan_reference_downloads,
    references_from_spec,
)
from bspp.orchestration.runtime.inputs.reports import report_to_json, report_to_text

__all__ = [
    "ArchiveCoverage",
    "ArchiveStagingItem",
    "ArchiveStagingPlan",
    "DownloadPlan",
    "ReferenceArtifact",
    "ReferenceStatus",
    "archives_for_dataset",
    "archives_for_runspec",
    "archives_for_runspec_staging",
    "archives_from_staging_dir",
    "check_references",
    "ensure_references",
    "plan_archive_staging",
    "plan_reference_downloads",
    "references_from_spec",
    "report_to_json",
    "report_to_text",
    "select_archives_for_array",
]
