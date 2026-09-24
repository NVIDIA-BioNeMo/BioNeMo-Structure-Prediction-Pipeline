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

"""S3 inventory evidence helpers for S3 ``s5cmd`` workflows."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.runtime.data_movement.common import PlannedTransfer
from bspp.orchestration.runtime.data_movement.s3.client import S3_ENDPOINT_ENV, S3Credentials

_LS_LINE = re.compile(r"^\S+\s+\S+\s+(?P<size>\d+)\s+(?P<uri>s3://\S+)$")


@dataclass(frozen=True, slots=True)
class S3InventoryObject:
    """One object listed under an S3 prefix."""

    uri: str
    size_bytes: int

    def to_redacted_dict(self) -> dict[str, object]:
        return {"uri": self.uri, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class SampleHashEvidence:
    """Hash evidence from a sampled remote-object download."""

    uri: str
    sha256: str
    size_bytes: int | None = None

    def to_redacted_dict(self) -> dict[str, object]:
        return {"uri": self.uri, "sha256": self.sha256, "size_bytes": self.size_bytes}


def build_list_prefix_argv(
    prefix: str,
    *,
    credentials: S3Credentials,
    s5cmd_path: str = "s5cmd",
    numworkers: int | None = None,
) -> tuple[str, ...]:
    """Build ``s5cmd ls`` argv for a prefix inventory collection."""
    argv = _base_argv(s5cmd_path, credentials, numworkers)
    argv.extend(["ls", _ensure_recursive_prefix(prefix)])
    return tuple(argv)


def build_sample_download_argv(
    uri: str,
    destination: Path,
    *,
    credentials: S3Credentials,
    s5cmd_path: str = "s5cmd",
    numworkers: int | None = None,
) -> tuple[str, ...]:
    """Build ``s5cmd cp`` argv for a single sampled object download."""
    argv = _base_argv(s5cmd_path, credentials, numworkers)
    argv.extend(["cp", uri, str(destination)])
    return tuple(argv)


def plan_list_prefix(
    prefix: str,
    *,
    credentials: S3Credentials,
    s5cmd_path: str = "s5cmd",
    numworkers: int | None = None,
) -> PlannedTransfer:
    """Return a dry-run transfer plan for prefix inventory collection."""
    return PlannedTransfer(
        tool="s5cmd",
        argv=build_list_prefix_argv(prefix, credentials=credentials, s5cmd_path=s5cmd_path, numworkers=numworkers),
        note=f"list prefix inventory (endpoint via {S3_ENDPOINT_ENV})",
    )


def parse_s5cmd_ls_output(output: str) -> tuple[S3InventoryObject, ...]:
    """Parse ``s5cmd ls`` output into object inventory rows."""
    objects: list[S3InventoryObject] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if " DIR " in f" {stripped} ":
            continue
        match = _LS_LINE.match(stripped)
        if match is None:
            msg = f"unparseable s5cmd ls line {line_number}: {line!r}"
            raise ValueError(msg)
        objects.append(S3InventoryObject(uri=match.group("uri"), size_bytes=int(match.group("size"))))
    return tuple(objects)


def read_inventory_csv(path: Path) -> tuple[S3InventoryObject, ...]:
    """Read inventory evidence CSV with ``uri,size_bytes`` columns."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames, {"uri", "size_bytes"})
        return tuple(
            S3InventoryObject(
                uri=_required(row, "uri", path=path, row_number=row_number),
                size_bytes=_read_int(row, "size_bytes", path=path, row_number=row_number),
            )
            for row_number, row in enumerate(reader, start=2)
        )


def write_inventory_csv(path: Path, objects: tuple[S3InventoryObject, ...]) -> Path:
    """Write inventory evidence CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("uri", "size_bytes"))
        writer.writeheader()
        for obj in objects:
            writer.writerow({"uri": obj.uri, "size_bytes": obj.size_bytes})
    tmp_path.replace(path)
    return path


def read_sample_hashes_csv(path: Path) -> tuple[SampleHashEvidence, ...]:
    """Read sampled remote hash evidence CSV."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, reader.fieldnames, {"uri", "sha256"})
        rows: list[SampleHashEvidence] = []
        for row_number, row in enumerate(reader, start=2):
            size_value = (row.get("size_bytes") or "").strip()
            rows.append(
                SampleHashEvidence(
                    uri=_required(row, "uri", path=path, row_number=row_number),
                    sha256=_required(row, "sha256", path=path, row_number=row_number),
                    size_bytes=int(size_value) if size_value else None,
                )
            )
        return tuple(rows)


def write_sample_hashes_csv(path: Path, hashes: tuple[SampleHashEvidence, ...]) -> Path:
    """Write sampled remote hash evidence CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("uri", "sha256", "size_bytes"))
        writer.writeheader()
        for item in hashes:
            writer.writerow({"uri": item.uri, "sha256": item.sha256, "size_bytes": item.size_bytes or ""})
    tmp_path.replace(path)
    return path


def _base_argv(s5cmd_path: str, credentials: S3Credentials, numworkers: int | None) -> list[str]:
    argv = [s5cmd_path, "--endpoint-url", credentials.endpoint_url]
    if numworkers is not None and numworkers > 0:
        argv.extend(["--numworkers", str(numworkers)])
    return argv


def _ensure_recursive_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("*") else f"{prefix.rstrip('/')}/*"


def _require_columns(path: Path, fieldnames: Sequence[str] | None, required: set[str]) -> None:
    if fieldnames is None:
        msg = f"CSV has no header: {path}"
        raise ValueError(msg)
    missing = required - set(fieldnames)
    if missing:
        msg = f"{path} is missing required columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)


def _required(row: dict[str, str], column: str, *, path: Path, row_number: int) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        msg = f"{path} row {row_number} has empty {column}"
        raise ValueError(msg)
    return value


def _read_int(row: dict[str, str], column: str, *, path: Path, row_number: int) -> int:
    value = _required(row, column, path=path, row_number=row_number)
    try:
        return int(value)
    except ValueError as exc:
        msg = f"{path} row {row_number} has invalid integer {column}: {value!r}"
        raise ValueError(msg) from exc


__all__ = [
    "S3InventoryObject",
    "SampleHashEvidence",
    "build_list_prefix_argv",
    "build_sample_download_argv",
    "parse_s5cmd_ls_output",
    "plan_list_prefix",
    "read_inventory_csv",
    "read_sample_hashes_csv",
    "write_inventory_csv",
    "write_sample_hashes_csv",
]
