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

"""Closed direct-style Database Placement Result facade."""

from __future__ import annotations

from collections.abc import Mapping

from bspp.orchestration.contract.database_capacity_fallback_result import (
    DatabaseCapacityFallbackResult,
    canonical_database_capacity_fallback_result_bytes,
    database_capacity_fallback_result_digest,
    database_capacity_fallback_result_from_mapping,
)
from bspp.orchestration.contract.database_placement_result import (
    DatabasePlacementResult,
    canonical_database_placement_result_bytes,
    database_placement_result_digest,
    database_placement_result_from_mapping,
)

DatabaseDirectResult = DatabasePlacementResult | DatabaseCapacityFallbackResult


def database_direct_result_from_mapping(payload: Mapping[str, object]) -> DatabaseDirectResult:
    """Dispatch one exact closed direct-style Result envelope."""
    if set(payload) == {"database_placement_result"}:
        return database_placement_result_from_mapping(payload)
    if set(payload) == {"database_capacity_fallback_result"}:
        return database_capacity_fallback_result_from_mapping(payload)
    raise ValueError("Database direct Result requires one known exact envelope")


def canonical_database_direct_result_bytes(result: DatabaseDirectResult) -> bytes:
    if isinstance(result, DatabasePlacementResult):
        return canonical_database_placement_result_bytes(result)
    if isinstance(result, DatabaseCapacityFallbackResult):
        return canonical_database_capacity_fallback_result_bytes(result)
    raise TypeError("unsupported Database direct Result type")


def database_direct_result_digest(result: DatabaseDirectResult) -> str:
    if isinstance(result, DatabasePlacementResult):
        return database_placement_result_digest(result)
    if isinstance(result, DatabaseCapacityFallbackResult):
        return database_capacity_fallback_result_digest(result)
    raise TypeError("unsupported Database direct Result type")


__all__ = [
    "DatabaseDirectResult",
    "canonical_database_direct_result_bytes",
    "database_direct_result_digest",
    "database_direct_result_from_mapping",
]
