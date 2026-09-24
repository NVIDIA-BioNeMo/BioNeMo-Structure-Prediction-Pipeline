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

"""Physical site composition for private Database Placement coordinators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.database_placement import (
    DATABASE_CACHE_ROOT,
    DATABASE_REPLICA_LEASE_TARGET,
    DATABASE_SOURCE_ROOT,
    SELECTED_DATABASE_ROOT,
)


@dataclass(frozen=True)
class DatabasePlacementPaths:
    """All container paths whose authority is fixed by production composition."""

    source_root: Path
    cache_root: Path
    selected_root: Path
    lease: Path
    mountinfo: Path
    rsync: Path


PRODUCTION_DATABASE_PLACEMENT_PATHS = DatabasePlacementPaths(
    source_root=Path(DATABASE_SOURCE_ROOT),
    cache_root=Path(DATABASE_CACHE_ROOT),
    selected_root=Path(SELECTED_DATABASE_ROOT),
    lease=Path(DATABASE_REPLICA_LEASE_TARGET),
    mountinfo=Path("/proc/self/mountinfo"),
    rsync=Path("/usr/bin/rsync"),
)


__all__ = ["PRODUCTION_DATABASE_PLACEMENT_PATHS", "DatabasePlacementPaths"]
