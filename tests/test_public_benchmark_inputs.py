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

"""Offline integrity and CLI compatibility of the public benchmark input release."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from bspp.orchestration.control.folding_benchmark.curator import _dataset_fingerprint, _parse_target_record
from bspp.orchestration.control.folding_benchmark.spec import load_benchmark_spec

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs/benchmarks/pdb-temporal-2022-2025-v1"
EXPECTED_FINGERPRINT = "c04ec62e6eecf165eea010f82f7aa1ad72ddfd99ce462a9c4afeac286049a217"
RELEASE_FILES = {
    "DATASET-LICENSE.md",
    "README.md",
    "assembly-urls.txt",
    "benchmark-provenance.json",
    "benchmark-spec.json",
    "benchmark-targets.csv",
    "full-reconstruction-verification.json",
    "pdb-ids.txt",
    "reconstruction-targets.jsonl",
    "verification.json",
}


def _target_rows() -> list[dict[str, str]]:
    with (BUNDLE / "benchmark-targets.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_public_benchmark_bundle_has_complete_checksums() -> None:
    lines = (BUNDLE / "SHA256SUMS").read_text().splitlines()
    entries = [line.split("  ", 1) for line in lines]
    assert len(entries) == len(RELEASE_FILES)
    assert {name for _, name in entries} == RELEASE_FILES
    assert {path.name for path in BUNDLE.iterdir()} == RELEASE_FILES | {"SHA256SUMS"}
    for digest, name in entries:
        path = BUNDLE / name
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_public_benchmark_records_match_reconstruction_contract() -> None:
    rows = _target_rows()
    records = [json.loads(line) for line in (BUNDLE / "reconstruction-targets.jsonl").read_text().splitlines()]
    spec = load_benchmark_spec(BUNDLE / "benchmark-spec.json")
    canonical_spec = load_benchmark_spec(ROOT / "configs/benchmark.pdb-temporal-v1.json")
    assert spec.raw_specification == canonical_spec.raw_specification
    assert len(rows) == len(records) == 1000
    assert len({row["target_id"] for row in rows}) == 1000
    assert len({row["pdb_id"] for row in rows}) == 1000
    assert [row["target_id"] for row in rows] == sorted(row["target_id"] for row in rows)
    assert [row["pdb_id"] for row in rows] == (BUNDLE / "pdb-ids.txt").read_text().splitlines()
    assert [row["source_url"] for row in rows] == (BUNDLE / "assembly-urls.txt").read_text().splitlines()
    strata = {stratum.name: stratum for stratum in spec.strata}
    for row, record in zip(rows, records, strict=True):
        parsed = _parse_target_record(record)
        assert parsed["target_id"] == row["target_id"]
        assert parsed["pdb_id"] == row["pdb_id"]
        assert parsed["assembly_id"] == row["assembly_id"] == "1"
        assert row["source_url"] == f"https://files.rcsb.org/download/{row['pdb_id'].lower()}-assembly1.cif.gz"
        assert parsed["stratum"] == row["stratum"]
        assert parsed["release_date"] == row["release_date"]
        chains = parsed["chains"]
        lengths = [len(chain) for chain in chains]
        assert lengths == parsed["chain_lengths"] == [int(value) for value in row["chain_lengths"].split(";")]
        assert sum(lengths) == parsed["total_length"] == int(row["total_residues"])
        assert all(set(chain) <= set("ACDEFGHIKLMNPQRSTVWY") for chain in chains)
        digest = hashlib.sha256(":".join(chains).encode()).hexdigest()
        assert digest == parsed["sequence_sha256"] == row["sequence_sha256"]
        stratum = strata[row["stratum"]]
        assert len(chains) == int(row["chain_count"]) == stratum.chain_count
        assert stratum.minimum_total_residues <= sum(lengths) <= stratum.maximum_total_residues
    assert Counter(row["stratum"] for row in rows) == {item.name: item.count for item in spec.strata}
    assert Counter(int(row["chain_count"]) for row in rows) == {1: 750, 2: 150, 3: 50, 4: 50}
    assert sum(int(row["total_residues"]) for row in rows) == 391180


def test_public_benchmark_pins_reproduce_published_identity() -> None:
    rows = _target_rows()
    spec = load_benchmark_spec(BUNDLE / "benchmark-spec.json")
    fingerprint = _dataset_fingerprint(
        spec.raw_specification,
        (
            (
                row["target_id"],
                row["sequence_sha256"],
                row["stratum"],
                row["reference_sha256"],
                row["source_mmcif_sha256"],
            )
            for row in rows
        ),
        source_key="source_mmcif_sha256",
        fingerprint_prefix=b"afdb-pdb-benchmark-mmcif-v1\0",
    )
    provenance = json.loads((BUNDLE / "benchmark-provenance.json").read_text())
    assert fingerprint == provenance["expected_dataset_fingerprint"] == EXPECTED_FINGERPRINT
    assert provenance["dataset_id"] == spec.dataset_id
    assert provenance["targets"] == len(rows)
    assert provenance["strata"] == Counter(row["stratum"] for row in rows)
    assert provenance["total_residues"] == sum(int(row["total_residues"]) for row in rows)
