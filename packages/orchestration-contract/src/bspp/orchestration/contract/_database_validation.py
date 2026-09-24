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

"""Shared strict parsing primitives for database contract consumers.

The helpers in this module deliberately preserve the different compatibility
policies of the public loaders.  In particular, placement paths retain their
historical NUL acceptance while replica-facts and provisioning paths reject
NUL bytes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from bspp.orchestration.contract.versioning import validate_schema_version


def require_fields(
    payload: Mapping[str, object],
    required: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> None:
    """Require the exact declared field closure, including permitted optionals."""
    actual = set(payload)
    allowed = required | (optional or set())
    if not required <= actual or not actual <= allowed:
        missing = sorted(required - actual)
        unknown = sorted(actual - allowed)
        raise ValueError(f"{name} fields are invalid; missing={missing!r}; unknown={unknown!r}")


def required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def required_sequence(payload: Mapping[str, object], key: str) -> list[object] | tuple[object, ...]:
    value = payload.get(key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"{key} must be a list")
    return value


def required_list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def required_mapping_sequence(payload: Mapping[str, object], key: str) -> tuple[Mapping[str, object], ...]:
    value = required_sequence(payload, key)
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{key}[{index}] must be a mapping")
        result.append(item)
    return tuple(result)


def required_str(payload: Mapping[str, object], key: str) -> str:
    """Extract a string while allowing empty values for provisioning compatibility."""
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def required_nonempty_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def required_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = required_sequence(payload, key)
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return tuple(value)  # type: ignore[arg-type]


def validate_placement_absolute_path(value: str, name: str) -> None:
    """Validate a placement-compatible POSIX path (including historical NUL acceptance)."""
    path = PurePosixPath(value)
    if not value or not path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"{name} must be a normalized absolute path")


def validate_replica_absolute_path(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if not value or "\x00" in value or not path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        raise ValueError(f"{name} must be a normalized absolute path")


def validate_provisioning_absolute_path(value: str, name: str) -> None:
    if not value or "\x00" in value or not Path(value).is_absolute() or os.path.normpath(value) != value:
        raise ValueError(f"{name} must be a normalized absolute path")


def validate_relative_path(value: str, name: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(item in {"", ".", ".."} for item in path.parts)
    ):
        raise ValueError(f"{name} must be a normalized confined relative path")


def validate_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def validate_mount_options(value: tuple[str, ...], name: str) -> None:
    if (
        not isinstance(value, tuple)
        or not value
        or tuple(sorted(set(value))) != value
        or any(not item or "," in item or "\x00" in item for item in value)
    ):
        raise ValueError(f"{name} must be a sorted unique immutable tuple")


def validate_database_source_mount_options(value: tuple[str, ...], name: str) -> None:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, str) or not item or "," in item or "\x00" in item for item in value)
        or tuple(sorted(set(value))) != value
    ):
        raise ValueError(f"database source {name} must be a sorted unique immutable tuple")


def validate_schema(value: int, name: str) -> None:
    if validate_schema_version(value, record_name=name) != value:
        raise ValueError(f"{name} schema_version must be declared explicitly")


__all__: Sequence[str] = ()
