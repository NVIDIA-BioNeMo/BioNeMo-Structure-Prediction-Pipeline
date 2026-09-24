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

"""Shard configuration generation and I/O.

Computes how many SLURM array shards are needed for a given number of
models (or archives) and writes/reads the resulting JSON config.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from bspp.orchestration.runtime.constants import MAX_PROTEINS_PER_SHARD

__all__ = [
    "ShardConfig",
    "compute_archive_shard_config",
    "compute_archive_shard_config_model",
    "compute_shard_config",
    "compute_shard_config_model",
    "load_shard_config",
    "read_shard_config",
    "write_shard_config",
]


class ShardConfig(BaseModel):
    """Typed representation of ``shard_config.json``.

    Flat-file and archive mode use the same wire file with mode-specific
    fields, so optional fields are omitted from compatibility dumps.
    """

    model_config = ConfigDict(extra="allow")

    total_models: int | None = None
    max_per_shard: int | None = None
    archive_mode: bool | None = None
    total_archives: int | None = None
    shards_per_archive: int | None = None
    required_shards: int
    array_range: str

    @field_validator("total_models", "total_archives", "required_shards", mode="before")
    @classmethod
    def _validate_non_negative_int(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            msg = f"must be a non-negative integer, got {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("max_per_shard", "shards_per_archive", mode="before")
    @classmethod
    def _validate_positive_int(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            msg = f"must be a positive integer, got {value!r}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_semantics(self) -> ShardConfig:
        expected_range = "0-0" if self.required_shards == 0 else f"0-{self.required_shards - 1}"
        if self.array_range != expected_range:
            msg = f"array_range must be {expected_range!r} for required_shards={self.required_shards}"
            raise ValueError(msg)
        if self.archive_mode:
            if self.total_archives != self.required_shards:
                msg = "archive mode requires total_archives == required_shards"
                raise ValueError(msg)
            if self.shards_per_archive is None:
                msg = "archive mode requires shards_per_archive"
                raise ValueError(msg)
        elif self.total_models is not None and self.max_per_shard is not None:
            expected_required = max(1, math.ceil(self.total_models / self.max_per_shard))
            if self.required_shards != expected_required:
                msg = f"required_shards must be {expected_required} for total_models/max_per_shard"
                raise ValueError(msg)
        return self

    def to_compat_dict(self) -> dict[str, Any]:
        """Return the JSON object shape used by existing callers."""
        return self.model_dump(exclude_none=True, exclude_unset=True)


def compute_shard_config_model(
    total_models: int,
    max_per_shard: int = MAX_PROTEINS_PER_SHARD,
) -> ShardConfig:
    """Compute typed shard parameters for flat-file mode."""
    if total_models < 0:
        msg = f"total_models must be non-negative, got {total_models}"
        raise ValueError(msg)
    if max_per_shard <= 0:
        msg = f"max_per_shard must be positive, got {max_per_shard}"
        raise ValueError(msg)

    required_shards = max(1, math.ceil(total_models / max_per_shard))
    return ShardConfig(
        total_models=total_models,
        max_per_shard=max_per_shard,
        required_shards=required_shards,
        array_range=f"0-{required_shards - 1}",
    )


def compute_shard_config(
    total_models: int,
    max_per_shard: int = MAX_PROTEINS_PER_SHARD,
) -> dict[str, Any]:
    """Compute shard parameters for flat-file (non-archive) mode.

    Returns a dict with keys: ``total_models``, ``max_per_shard``,
    ``required_shards``, ``array_range``.
    """
    return compute_shard_config_model(total_models, max_per_shard).to_compat_dict()


def compute_archive_shard_config_model(
    archives: list[str],
    shards_per_archive: int = 2,
) -> ShardConfig:
    """Compute typed shard parameters for archive mode."""
    if shards_per_archive <= 0:
        msg = f"shards_per_archive must be positive, got {shards_per_archive}"
        raise ValueError(msg)

    n_archives = len(archives)
    required_shards = n_archives
    return ShardConfig(
        archive_mode=True,
        total_archives=n_archives,
        shards_per_archive=shards_per_archive,
        required_shards=required_shards,
        array_range=f"0-{required_shards - 1}" if required_shards > 0 else "0-0",
    )


def compute_archive_shard_config(
    archives: list[str],
    shards_per_archive: int = 2,
) -> dict[str, Any]:
    """Compute shard parameters for archive mode.

    Each archive becomes one SLURM task; *shards_per_archive* controls how
    many sub-shards are processed sequentially within each task.

    Returns a dict with keys: ``archive_mode``, ``total_archives``,
    ``shards_per_archive``, ``required_shards``, ``array_range``.
    """
    return compute_archive_shard_config_model(archives, shards_per_archive).to_compat_dict()


def write_shard_config(config: Mapping[str, Any] | ShardConfig, output_path: Path) -> None:
    """Write *config* as pretty-printed JSON to *output_path* atomically.

    The earlier in-place ``write_text`` opened the destination O_TRUNC,
    so a reader that landed mid-rewrite would observe an empty or
    truncated file. Combined with the fail-closed
    :class:`~bspp.orchestration.runtime.postprocessing.sharding.InvalidShardConfigError`
    behavior in :func:`~bspp.orchestration.runtime.postprocessing.sharding.read_required_shards`,
    that converted a transient rewrite window into a hard abort for
    every concurrent ``upload-s3`` / ``aggregate`` / ``validate
    coverage`` invocation.

    Use the standard ``write tempfile -> fsync -> os.replace`` dance so
    readers always see either the previous complete file or the new
    complete file — never a partial one. ``os.replace`` is atomic on
    POSIX and on Windows (Python ≥ 3.3).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_obj = config.to_compat_dict() if isinstance(config, ShardConfig) else dict(config)
    payload = json.dumps(payload_obj, indent=2) + "\n"
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with open(tmp_path, "w") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, output_path)


def read_shard_config(path: Path) -> dict[str, Any]:
    """Read a shard config JSON file and return its contents."""
    result: dict[str, Any] = json.loads(path.read_text())
    return result


def load_shard_config(path: Path) -> ShardConfig:
    """Read and validate a shard config JSON file as a typed model."""
    return ShardConfig.model_validate(read_shard_config(path))
