#!/usr/bin/env python3
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

"""Parse the supervision contract from the baked carry-characterization helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


class SupervisionError(ValueError):
    """Raised when helper bytes do not expose one unambiguous supervision contract."""


_WARMUP = re.compile(r"warmup_second\s*<\s*(\d+)")
_TIMEOUT = re.compile(r"--kill-after=(\d+)s\s+(\d+)s")
_THREADS = re.compile(r"--threads\s+(\d+)")
_KEYS = frozenset({"gpuserver_warmup_seconds", "search_timeout_seconds", "search_kill_after_seconds", "search_threads"})


def parse_helper_bytes(data: bytes) -> dict[str, int]:
    """Return the single supervision declaration encoded in a helper shell script."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SupervisionError("baked helper is not UTF-8") from exc
    warmups = _WARMUP.findall(text)
    timeouts = _TIMEOUT.findall(text)
    threads = _THREADS.findall(text)
    if len(warmups) != len(timeouts) or len(timeouts) != len(threads) or len(warmups) != 1:
        raise SupervisionError("helper must contain exactly one warmup, timeout, and threads declaration")
    kill_after, timeout = timeouts[0]
    values = {
        "gpuserver_warmup_seconds": int(warmups[0]),
        "search_timeout_seconds": int(timeout),
        "search_kill_after_seconds": int(kill_after),
        "search_threads": int(threads[0]),
    }
    if any(value <= 0 for value in values.values()):
        raise SupervisionError("helper supervision values must be positive")
    return values


def parse_helper_file(path: Path) -> dict[str, int]:
    return parse_helper_bytes(path.read_bytes())


def validate_manifest(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _KEYS:
        raise SupervisionError("search_supervision has an unexpected key set")
    parsed: dict[str, int] = {}
    for key in _KEYS:
        item = value[key]
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise SupervisionError(f"search_supervision.{key} must be a positive integer")
        parsed[key] = item
    return parsed


def manifest_with_helper_sha256(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    with Path(__file__).resolve(strict=True).open("rb") as parser_source:
        parser_sha256 = hashlib.file_digest(parser_source, "sha256").hexdigest()
    return {
        "helper_sha256": hashlib.sha256(data).hexdigest(),
        "supervision_parser_sha256": parser_sha256,
        "search_supervision": parse_helper_bytes(data),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("helper", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = manifest_with_helper_sha256(args.helper)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
