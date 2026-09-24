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

"""Compatibility facade for focused Database Replica lock modules."""

from __future__ import annotations

from ._database_replica_lock_authority import (
    acquire_existing_shared_identity_lock,
    acquire_or_create_exclusive_population_lock,
    verify_cache_ownership,
    verify_existing_shared_identity_lock,
    verify_identity_ownership,
    verify_maintenance_cache_ownership,
    verify_maintenance_identity_ownership,
    verify_shared_identity_ownership,
)
from ._database_replica_lock_coordination import (
    hold_exclusive_cache,
    hold_exclusive_cache_for_maintenance,
    hold_exclusive_identities_for_maintenance,
    hold_exclusive_identity,
)
from ._database_replica_lock_types import (
    CacheExclusiveOwnership,
    FilesystemAuthority,
    IdentityExclusiveOwnership,
    LockContendedError,
    LockWait,
    MaintenanceCacheExclusiveOwnership,
    MaintenanceIdentityContendedError,
    MaintenanceIdentityExclusiveOwnership,
)

__all__ = [
    "CacheExclusiveOwnership",
    "FilesystemAuthority",
    "IdentityExclusiveOwnership",
    "LockContendedError",
    "LockWait",
    "MaintenanceCacheExclusiveOwnership",
    "MaintenanceIdentityContendedError",
    "MaintenanceIdentityExclusiveOwnership",
    "acquire_existing_shared_identity_lock",
    "acquire_or_create_exclusive_population_lock",
    "hold_exclusive_cache",
    "hold_exclusive_cache_for_maintenance",
    "hold_exclusive_identities_for_maintenance",
    "hold_exclusive_identity",
    "verify_cache_ownership",
    "verify_existing_shared_identity_lock",
    "verify_identity_ownership",
    "verify_maintenance_cache_ownership",
    "verify_maintenance_identity_ownership",
    "verify_shared_identity_ownership",
]
