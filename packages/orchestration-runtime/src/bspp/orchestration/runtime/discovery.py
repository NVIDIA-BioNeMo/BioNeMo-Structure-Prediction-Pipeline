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

"""Discover model IDs from an input directory by scanning filenames.

Performs a single os.scandir() pass, strips known suffixes to extract model
IDs, deduplicates, sorts, and returns the list. Deterministic output
guarantees repeatable sharding across SLURM array tasks.

Works on both canonical (AF-XXXX-model_v1.pdb) and ColabFold
(AF_XXXX_AF_YYYY.merged_unrelaxed_...pdb) naming conventions.
"""

from __future__ import annotations

import os
from pathlib import Path

from bspp.orchestration.runtime.constants import KNOWN_SUFFIXES

__all__ = [
    "build_file_index",
    "discover",
    "extract_model_id",
]

_AFDB_PREFIX = "AFDB_"


def extract_model_id(filename: str) -> str | None:
    """Extract model_id from a filename by stripping known suffixes.

    Also strips the ``AFDB_`` prefix that ColabFold batch outputs prepend,
    so ``AFDB_AF_0000...`` normalises to ``AF_0000...`` which matches the
    manifest's ``AF-0000...`` convention (the resolver handles ``_`` vs ``-``).
    """
    for suffix in KNOWN_SUFFIXES:
        if filename.endswith(suffix):
            model_id = filename[: -len(suffix)]
            if model_id.startswith(_AFDB_PREFIX):
                model_id = model_id[len(_AFDB_PREFIX) :]
            return model_id
    return None


def discover(input_dir: Path) -> list[str]:
    """Scan *input_dir* and return a sorted, deduplicated list of model IDs."""
    model_ids: set[str] = set()
    for entry in os.scandir(input_dir):
        if not entry.is_file():
            continue
        mid = extract_model_id(entry.name)
        if mid is not None:
            model_ids.add(mid)
    return sorted(model_ids)


def build_file_index(input_dir: Path) -> dict[str, list[str]]:
    """Scan *input_dir* once and return ``{model_id: [filename, ...]}``.

    The returned index lets downstream shards construct full paths without
    re-scanning the (potentially huge) directory.
    """
    index: dict[str, list[str]] = {}
    for entry in os.scandir(input_dir):
        if not entry.is_file():
            continue
        mid = extract_model_id(entry.name)
        if mid is not None:
            index.setdefault(mid, []).append(entry.name)
    return index
