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

"""Typed cold Database Replica failures for exact evidence classification."""

from __future__ import annotations

from bspp.orchestration.contract.database_replica import DatabaseReplicaColdFailureClassification

from ._database_placement_errors import DatabasePlacementError


class ClassifiedDatabaseReplicaError(DatabasePlacementError):
    """A cold placement error carrying its stable bounded classification."""

    def __init__(self, classification: DatabaseReplicaColdFailureClassification, message: str) -> None:
        super().__init__(message)
        self.classification = classification


__all__ = ["ClassifiedDatabaseReplicaError"]
