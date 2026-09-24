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

"""Benchmark corpus specification model and JSON loader.

Pure-stdlib port of the benchmark spec model from the frozen reference pipeline
harvest source (``src/afdb_pipeline/benchmark_dataset.py``), reduced to
the fields the folding benchmark curator needs. ``source`` and ``filters`` are kept
opaque (validated only as objects) so the verbatim temporal-holdout config loads
without this package depending on the curator's RCSB query semantics.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkStratum:
    """One residue-range stratum of the benchmark corpus."""

    name: str
    chain_count: int
    minimum_total_residues: int
    maximum_total_residues: int
    count: int


@dataclass(frozen=True)
class BenchmarkSpec:
    """Parsed benchmark corpus specification."""

    schema_version: int
    dataset_id: str
    selection_seed: str
    source: Mapping[str, object]  # opaque RCSB source filters
    filters: Mapping[str, object]  # opaque scientific filters
    strata: tuple[BenchmarkStratum, ...]
    throughput_subset_sizes: tuple[int, ...]
    description: str
    raw_specification: Mapping[str, object]  # verbatim parsed document for fingerprinting


def _require_integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _parse_stratum(item: Mapping[Any, Any], index: int, names: set[str]) -> BenchmarkStratum:
    name = _require_non_empty_string(item.get("name"), f"strata[{index}].name")
    if name in names:
        raise ValueError(f"strata[{index}].name must be unique")
    names.add(name)
    chain_count = _require_integer(item.get("chain_count"), f"{name}.chain_count", minimum=1)
    count = _require_integer(item.get("count"), f"{name}.count", minimum=1)

    has_total = "total_residues" in item
    has_min = "minimum_total_residues" in item
    has_max = "maximum_total_residues" in item
    if has_total and (has_min or has_max):
        raise ValueError(f"{name} must not mix total_residues with minimum/maximum_total_residues")
    if has_total:
        total = _require_integer(item.get("total_residues"), f"{name}.total_residues", minimum=1)
        minimum = total
        maximum = total
    else:
        minimum = _require_integer(item.get("minimum_total_residues"), f"{name}.minimum_total_residues", minimum=1)
        maximum = _require_integer(
            item.get("maximum_total_residues"), f"{name}.maximum_total_residues", minimum=minimum
        )

    return BenchmarkStratum(
        name=name,
        chain_count=chain_count,
        minimum_total_residues=minimum,
        maximum_total_residues=maximum,
        count=count,
    )


def load_benchmark_spec(path: Path) -> BenchmarkSpec:
    """Parse and validate a benchmark specification JSON document.

    Unknown top-level keys (e.g. the config's ``description`` or extension
    fields) are retained verbatim in ``raw_specification`` so the curator can
    fingerprint and publish the complete authored specification rather than a
    lossy normalized projection. A missing top-level ``description`` is
    normalized to ``""`` for backward compatibility; a present non-string
    description is rejected. All validation failures raise :class:`ValueError`.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read benchmark specification {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("benchmark specification must be a JSON object")
    schema_version = raw.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError("benchmark specification schema_version must be 1")
    dataset_id = _require_non_empty_string(raw.get("dataset_id"), "dataset_id")
    selection_seed = _require_non_empty_string(raw.get("selection_seed"), "selection_seed")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError("benchmark specification description must be a string")

    source = raw.get("source")
    filters = raw.get("filters")
    if not isinstance(source, dict) or not isinstance(filters, dict):
        raise ValueError("benchmark source and filters must be objects")

    strata_value = raw.get("strata")
    subset_sizes = raw.get("throughput_subset_sizes")
    if not isinstance(strata_value, list) or not strata_value:
        raise ValueError("benchmark strata must be a non-empty list")
    if not isinstance(subset_sizes, list) or not subset_sizes:
        raise ValueError("benchmark throughput_subset_sizes must be a non-empty list")

    names: set[str] = set()
    strata: list[BenchmarkStratum] = []
    for index, item in enumerate(strata_value):
        if not isinstance(item, dict):
            raise ValueError(f"strata[{index}] must be an object")
        strata.append(_parse_stratum(item, index, names))

    parsed_subsets = tuple(_require_integer(value, "throughput_subset_sizes item", minimum=1) for value in subset_sizes)

    return BenchmarkSpec(
        schema_version=1,
        dataset_id=dataset_id,
        selection_seed=selection_seed,
        source=source,
        filters=filters,
        strata=tuple(strata),
        throughput_subset_sizes=parsed_subsets,
        description=description,
        raw_specification=raw,
    )
