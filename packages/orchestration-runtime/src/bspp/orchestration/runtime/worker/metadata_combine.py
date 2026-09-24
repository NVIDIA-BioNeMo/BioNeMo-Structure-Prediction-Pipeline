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

"""Orchestration-owned AF metadata combine helper."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson


class MetadataCombineError(RuntimeError):
    """Raised when metadata inputs cannot be combined."""


def iter_input_files(directory: Path, pattern: str = "*.json") -> tuple[Path, ...]:
    """Return sorted matching input files, matching legacy combine semantics."""

    files = tuple(path for path in sorted(directory.glob(pattern)) if path.is_file())
    if not files:
        msg = f"No files matching pattern {pattern!r} in {directory}"
        raise MetadataCombineError(msg)
    return files


def load_records(paths: Iterable[Path]) -> list[Any]:
    """Load and flatten records from metadata JSON files."""

    records: list[Any] = []
    for path in paths:
        data = orjson.loads(path.read_bytes())
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)
    return records


def combine_metadata_files(
    *,
    input_dir: Path,
    output_dir: Path,
    output_filename: str | None = None,
    output_prefix: str = "AF-metadata",
    chunk_size: int = 10_000,
    pattern: str = "*.json",
) -> tuple[Path, ...]:
    """Combine per-accession metadata JSONs into exact or chunked outputs."""

    files = iter_input_files(input_dir.resolve(), pattern)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_filename:
        records = load_records(files)
        path = output_dir / output_filename
        _write_records(path, records)
        return (path,)

    total_batches = (len(files) + chunk_size - 1) // chunk_size
    outputs: list[Path] = []
    for index, group in enumerate(_chunked(files, chunk_size), start=1):
        records = load_records(group)
        path = output_dir / f"{output_prefix}-{index}-of-{total_batches}.json"
        _write_records(path, records)
        outputs.append(path)
    return tuple(outputs)


def _chunked(paths: Iterable[Path], size: int) -> Iterable[list[Path]]:
    batch: list[Path] = []
    for path in paths:
        batch.append(path)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_records(path: Path, records: list[Any]) -> None:
    with path.open("wb") as handle:
        handle.write(orjson.dumps(records, option=orjson.OPT_INDENT_2))
        handle.write(b"\n")


__all__ = [
    "MetadataCombineError",
    "combine_metadata_files",
    "iter_input_files",
    "load_records",
]
