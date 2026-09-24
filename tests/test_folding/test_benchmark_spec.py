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

"""Tests for the folding benchmark corpus specification model and loader."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bspp.orchestration.control.folding_benchmark.spec import (
    BenchmarkStratum,
    load_benchmark_spec,
)


def _write_spec(tmp_path: Path, **overrides: object) -> Path:
    document: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": "pdb-temporal-2022-2025-v1",
        "selection_seed": "afdb-public-benchmark-v1",
        "source": {"provider": "RCSB PDB"},
        "filters": {"minimum_chain_length": 40},
        "strata": [
            {
                "name": "monomer_short",
                "chain_count": 1,
                "minimum_total_residues": 50,
                "maximum_total_residues": 149,
                "count": 200,
            }
        ],
        "throughput_subset_sizes": [10, 50, 100],
    }
    document.update(overrides)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_load_round_trip_range_form(tmp_path: Path) -> None:
    path = _write_spec(tmp_path)
    spec = load_benchmark_spec(path)
    assert spec.schema_version == 1
    assert spec.dataset_id == "pdb-temporal-2022-2025-v1"
    assert spec.selection_seed == "afdb-public-benchmark-v1"
    assert spec.source == {"provider": "RCSB PDB"}
    assert spec.filters == {"minimum_chain_length": 40}
    assert spec.strata == (
        BenchmarkStratum(
            name="monomer_short",
            chain_count=1,
            minimum_total_residues=50,
            maximum_total_residues=149,
            count=200,
        ),
    )
    assert spec.strata[0].count == 200
    assert spec.throughput_subset_sizes == (10, 50, 100)
    assert isinstance(spec.strata, tuple)
    assert isinstance(spec.throughput_subset_sizes, tuple)


def test_load_round_trip_total_residues_shorthand(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path,
        strata=[{"name": "monomer_short", "chain_count": 1, "total_residues": 100, "count": 200}],
    )
    spec = load_benchmark_spec(path)
    assert spec.strata[0].minimum_total_residues == 100
    assert spec.strata[0].maximum_total_residues == 100
    assert spec.strata[0].count == 200


@pytest.mark.parametrize(
    "missing_field",
    [
        "schema_version",
        "dataset_id",
        "selection_seed",
        "source",
        "filters",
        "strata",
        "throughput_subset_sizes",
    ],
)
def test_missing_field_rejected(tmp_path: Path, missing_field: str) -> None:
    document: dict[str, object] = {
        "schema_version": 1,
        "dataset_id": "pdb-temporal-2022-2025-v1",
        "selection_seed": "afdb-public-benchmark-v1",
        "source": {},
        "filters": {},
        "strata": [{"name": "monomer_short", "chain_count": 1, "total_residues": 100, "count": 200}],
        "throughput_subset_sizes": [10, 100],
    }
    del document[missing_field]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_spec(path)


def test_unknown_top_level_keys_tolerated(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, description="an ignored description", extra_key="ignored")
    spec = load_benchmark_spec(path)
    assert spec.dataset_id == "pdb-temporal-2022-2025-v1"


def test_top_level_description_survives_loading(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, description="a temporal-holdout benchmark")
    spec = load_benchmark_spec(path)
    assert spec.description == "a temporal-holdout benchmark"


def test_raw_specification_equals_parsed_document(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, description="kept verbatim", extra_key={"nested": [1, 2, 3]})
    spec = load_benchmark_spec(path)
    assert dict(spec.raw_specification) == json.loads(path.read_text(encoding="utf-8"))
    assert spec.raw_specification.get("extra_key") == {"nested": [1, 2, 3]}


def test_missing_description_is_empty_but_raw_mapping_preserves_absence(tmp_path: Path) -> None:
    path = _write_spec(tmp_path)
    spec = load_benchmark_spec(path)
    assert spec.description == ""
    assert "description" not in spec.raw_specification


def test_non_string_description_rejected(tmp_path: Path) -> None:
    path = _write_spec(tmp_path, description=42)
    with pytest.raises(ValueError, match="description must be a string"):
        load_benchmark_spec(path)


def test_non_positive_stratum_count_rejected(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path,
        strata=[{"name": "monomer_short", "chain_count": 1, "total_residues": 100, "count": 0}],
    )
    with pytest.raises(ValueError):
        load_benchmark_spec(path)


@pytest.mark.parametrize("bad_subset", [[0], [-1], [1, "two"]])
def test_bad_subset_size_rejected(tmp_path: Path, bad_subset: object) -> None:
    path = _write_spec(tmp_path, throughput_subset_sizes=bad_subset)
    with pytest.raises(ValueError):
        load_benchmark_spec(path)


def test_strata_is_frozen(tmp_path: Path) -> None:
    path = _write_spec(tmp_path)
    spec = load_benchmark_spec(path)
    with pytest.raises(FrozenInstanceError):
        spec.strata[0].count = 1  # type: ignore[misc]


def test_mixed_total_residues_and_range_rejected(tmp_path: Path) -> None:
    path = _write_spec(
        tmp_path,
        strata=[
            {
                "name": "monomer_short",
                "chain_count": 1,
                "total_residues": 100,
                "minimum_total_residues": 50,
                "maximum_total_residues": 149,
                "count": 200,
            }
        ],
    )
    with pytest.raises(ValueError):
        load_benchmark_spec(path)
