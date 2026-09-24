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

"""Deterministic JSON and text reports for input preparation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast


def report_to_json(report: object) -> str:
    """Serialize a report object as deterministic JSON."""
    return json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n"


def report_to_text(report: object) -> str:
    """Serialize a report object as compact deterministic text."""
    payload = _jsonable(report)
    lines: list[str] = []
    _append_text(lines, payload)
    return "\n".join(lines) + ("\n" if lines else "")


def write_json_report(report: object, path: Path) -> Path:
    """Write a report as deterministic JSON via atomic replace."""
    return _write_text(report_to_json(report), path)


def write_text_summary(report: object, path: Path) -> Path:
    """Write a compact deterministic text report via atomic replace."""
    return _write_text(report_to_text(report), path)


def _write_text(payload: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path


def _jsonable(value: object) -> object:
    if hasattr(value, "to_redacted_dict"):
        return _jsonable(cast(Any, value).to_redacted_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _append_text(lines: list[str], value: object, *, prefix: str = "") -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            item = value[key]
            if isinstance(item, dict | list):
                lines.append(f"{prefix}{key}:")
                _append_text(lines, item, prefix=f"{prefix}  ")
            else:
                lines.append(f"{prefix}{key}: {item}")
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                _append_text(lines, item, prefix=f"{prefix}  ")
            else:
                lines.append(f"{prefix}- {item}")
        return
    lines.append(f"{prefix}{value}")


__all__ = ["report_to_json", "report_to_text", "write_json_report", "write_text_summary"]
