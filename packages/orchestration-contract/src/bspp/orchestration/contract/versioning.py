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

"""Shared contract schema-version helpers."""

from __future__ import annotations

from typing import Literal

CURRENT_CONTRACT_SCHEMA_VERSION: Literal[1] = 1
SUPPORTED_CONTRACT_SCHEMA_VERSIONS = frozenset({CURRENT_CONTRACT_SCHEMA_VERSION})


class UnsupportedSchemaVersionError(ValueError):
    """Raised when a contract record declares an unsupported schema version."""


def validate_schema_version(value: object, *, record_name: str) -> int:
    """Return a supported schema version or fail closed."""
    if value is None:
        return CURRENT_CONTRACT_SCHEMA_VERSION
    if not isinstance(value, int) or isinstance(value, bool) or value not in SUPPORTED_CONTRACT_SCHEMA_VERSIONS:
        supported = ", ".join(str(version) for version in sorted(SUPPORTED_CONTRACT_SCHEMA_VERSIONS))
        msg = f"Unsupported {record_name} schema_version {value!r}; supported versions: {supported}"
        raise UnsupportedSchemaVersionError(msg)
    return value


__all__ = [
    "CURRENT_CONTRACT_SCHEMA_VERSION",
    "SUPPORTED_CONTRACT_SCHEMA_VERSIONS",
    "UnsupportedSchemaVersionError",
    "validate_schema_version",
]
