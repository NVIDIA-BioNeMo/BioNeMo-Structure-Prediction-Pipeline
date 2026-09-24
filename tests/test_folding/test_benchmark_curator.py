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

"""Tests for the folding benchmark dataset curator."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bspp.orchestration.control.folding_benchmark.curator import (
    DATA_API,
    FILES_URL,
    MAX_DECOMPRESSED_BYTES,
    SEARCH_URL,
    BenchmarkCurationFailed,
    CandidateRejected,
    _dataset_fingerprint,
    _gunzip_bounded,
    _ranked_throughput_order,
    prepare_benchmark_dataset,
)
from bspp.orchestration.control.folding_benchmark.spec import BenchmarkSpec, BenchmarkStratum


def _atom_line(serial: int, residue: str, chain: str, resnum: int, x: float, y: float, z: float) -> str:
    return f"ATOM  {serial:5d}  CA  {residue:>3} {chain}{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"


def _pdb_text(chains: Mapping[str, tuple[str, ...]]) -> str:
    lines: list[str] = []
    serial = 0
    for chain_id, residues in chains.items():
        lines.append(f"SEQRES   1 {chain_id} {len(residues):4d}  {' '.join(residues)}")
        for index, residue in enumerate(residues, start=1):
            serial += 1
            lines.append(_atom_line(serial, residue, chain_id, index, 10.0 + index, 20.0 + index, 30.0 + index))
    return "\n".join(lines) + "\n"


def _assembly(rcsb_candidate_assembly: str = "Y") -> dict[str, Any]:
    return {
        "pdbx_struct_assembly": {
            "rcsb_candidate_assembly": rcsb_candidate_assembly,
            "details": "author_and_software_defined_assembly",
        },
        "rcsb_assembly_info": {"polymer_composition": "Protein"},
    }


def _entry(
    *,
    polymer_entity_ids: list[str],
    release_date: str = "2023-05-15",
    resolution: float = 2.0,
    method: str = "X-RAY DIFFRACTION",
) -> dict[str, Any]:
    return {
        "rcsb_accession_info": {"initial_release_date": release_date},
        "exptl": [{"method": method}],
        "rcsb_entry_info": {"resolution_combined": [resolution]},
        "rcsb_entry_container_identifiers": {"polymer_entity_ids": polymer_entity_ids},
    }


def _entity(sequence: str, cluster_id: int) -> dict[str, Any]:
    return {
        "entity_poly": {"pdbx_seq_one_letter_code_can": sequence},
        "rcsb_cluster_membership": [{"identity": 30, "cluster_id": cluster_id}],
    }


# Two valid targets (one monomer, one dimer) plus one non-matching candidate that
# is rejected for chain count (dimer stratum) and length (monomer stratum).
PDB_TEXTS = {
    "1abc": _pdb_text({"A": ("ALA", "GLY")}),
    "2def": _pdb_text({"A": ("ALA",), "B": ("GLY",)}),
    "3bad": _pdb_text({"A": ("ALA", "GLY", "SER")}),
}

ASSEMBLIES = {
    "1abc": _assembly(),
    "2def": _assembly(),
    "3bad": _assembly(),
}

ENTRIES = {
    "1abc": _entry(polymer_entity_ids=["1"]),
    "2def": _entry(polymer_entity_ids=["1", "2"]),
    "3bad": _entry(polymer_entity_ids=["1"]),
}

ENTITIES = {
    ("1abc", "1"): _entity("AG", 100),
    ("2def", "1"): _entity("A", 200),
    ("2def", "2"): _entity("G", 300),
    ("3bad", "1"): _entity("AGS", 400),
}


def _make_spec() -> BenchmarkSpec:
    source = {
        "provider": "RCSB PDB",
        "release_date_min": "2022-01-01",
        "release_date_max": "2025-12-31",
        "max_resolution_angstrom": 3.0,
        "required_any_experimental_methods": ["X-RAY DIFFRACTION"],
        "assembly_id": "1",
    }
    filters = {
        "minimum_chain_length": 1,
        "minimum_coordinate_coverage": 0.5,
        "canonical_amino_acids_only": True,
        "require_candidate_biological_assembly": True,
        "monomer_sequence_cluster_identity": 30,
    }
    strata = (
        BenchmarkStratum(
            name="monomer_short",
            chain_count=1,
            minimum_total_residues=1,
            maximum_total_residues=2,
            count=1,
        ),
        BenchmarkStratum(
            name="dimer_small",
            chain_count=2,
            minimum_total_residues=1,
            maximum_total_residues=2,
            count=1,
        ),
    )
    return BenchmarkSpec(
        schema_version=1,
        dataset_id="pdb-temporal-2022-2025-v1",
        selection_seed="test-seed",
        source=source,
        filters=filters,
        strata=strata,
        throughput_subset_sizes=(10,),
        description="test-benchmark",
        raw_specification={
            "schema_version": 1,
            "dataset_id": "pdb-temporal-2022-2025-v1",
            "description": "test-benchmark",
            "selection_seed": "test-seed",
            "source": source,
            "filters": filters,
            "strata": [
                {
                    "name": "monomer_short",
                    "chain_count": 1,
                    "minimum_total_residues": 1,
                    "maximum_total_residues": 2,
                    "count": 1,
                },
                {
                    "name": "dimer_small",
                    "chain_count": 2,
                    "minimum_total_residues": 1,
                    "maximum_total_residues": 2,
                    "count": 1,
                },
            ],
            "throughput_subset_sizes": [10],
        },
    )


def _fake_request(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: int = 60,
    maximum_bytes: int | None = None,
) -> bytes:
    if url == SEARCH_URL:
        return json.dumps({"result_set": ["1abc-1", "2def-1", "3bad-1"]}).encode()
    if url.startswith(f"{DATA_API}/assembly/"):
        pdb_id = url.split("/")[-2].lower()
        return json.dumps(ASSEMBLIES[pdb_id]).encode()
    if url.startswith(f"{DATA_API}/entry/"):
        pdb_id = url.split("/")[-1].lower()
        return json.dumps(ENTRIES[pdb_id]).encode()
    if url.startswith(f"{DATA_API}/polymer_entity/"):
        parts = url.split("/")
        pdb_id = parts[-2].lower()
        entity_id = parts[-1]
        return json.dumps(ENTITIES[(pdb_id, entity_id)]).encode()
    if url.startswith(f"{FILES_URL}/"):
        pdb_id = url.split("/")[-1].split(".pdb", maxsplit=1)[0].lower()
        return gzip.compress(PDB_TEXTS[pdb_id].encode())
    raise AssertionError(f"unexpected request URL: {url}")


def _run_curation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: Path) -> dict[str, Any]:
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", _fake_request)
    return prepare_benchmark_dataset(output, _make_spec(), workers=2, progress=lambda _message: None)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_curation_materializes_full_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "benchmark"
    dataset = _run_curation(tmp_path, monkeypatch, output)

    assert dataset["targets"] == 2
    assert dataset["output"] == str(output.resolve())

    expected_files = [
        output / "dataset.json",
        output / "targets.fasta",
        output / "targets.jsonl",
        output / "targets.parquet",
        output / "validation-suite.json",
        output / "subsets" / "n0010" / "targets.fasta",
        output / "subsets" / "n0010" / "target-ids.jsonl",
        output / "validation-suite-n0010.json",
        output / "throughput-subsets.json",
        output / "SHA256SUMS",
        output / "DATASET-LICENSE.md",
        output / "references" / "pdb_1abc_assembly_1.pdb",
        output / "references" / "pdb_2def_assembly_1.pdb",
    ]
    for path in expected_files:
        assert path.is_file(), f"missing {path}"

    public = _read_jsonl(output / "targets.jsonl")
    target_ids = {record["target_id"] for record in public}
    assert target_ids == {"pdb_1abc_assembly_1", "pdb_2def_assembly_1"}

    # The non-matching candidate was rejected, never materialized.
    assert all("3bad" not in record["target_id"] for record in public)

    # Rejection accounting captured the CandidateRejected-driven filtering.
    dataset_json = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    rejections = dataset_json["selection"]["rejections"]
    assert rejections, "expected at least one candidate rejection"
    assert set(rejections) <= {
        "reference_chain_count_does_not_match_the_stratum",
        "reference_length_does_not_match_the_stratum",
    }


def test_dataset_fingerprint_is_stable_across_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_curation(tmp_path, monkeypatch, tmp_path / "first")
    _run_curation(tmp_path, monkeypatch, tmp_path / "second")

    first_json = json.loads((tmp_path / "first" / "dataset.json").read_text(encoding="utf-8"))
    second_json = json.loads((tmp_path / "second" / "dataset.json").read_text(encoding="utf-8"))

    fingerprint = first_json["dataset_fingerprint"]
    assert fingerprint == second_json["dataset_fingerprint"]
    assert len(fingerprint) == 64
    assert all(ch in "0123456789abcdef" for ch in fingerprint)
    assert first_json["dataset_id"] == "pdb-temporal-2022-2025-v1"


def test_n0010_subset_is_ranked_throughput_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "benchmark"
    _run_curation(tmp_path, monkeypatch, output)

    public = _read_jsonl(output / "targets.jsonl")
    expected_order = [record["target_id"] for record in _ranked_throughput_order(public, _make_spec().strata)]
    subset_ids = [record["target_id"] for record in _read_jsonl(output / "subsets" / "n0010" / "target-ids.jsonl")]
    assert subset_ids == expected_order[:10]
    assert set(subset_ids) == {"pdb_1abc_assembly_1", "pdb_2def_assembly_1"}


def test_ranked_throughput_order_round_robin() -> None:
    strata = (
        BenchmarkStratum(name="a", chain_count=1, minimum_total_residues=1, maximum_total_residues=10, count=2),
        BenchmarkStratum(name="b", chain_count=1, minimum_total_residues=1, maximum_total_residues=10, count=3),
    )
    records = [
        {"target_id": "a0", "stratum": "a", "selection_rank": 0},
        {"target_id": "a1", "stratum": "a", "selection_rank": 1},
        {"target_id": "b0", "stratum": "b", "selection_rank": 0},
        {"target_id": "b1", "stratum": "b", "selection_rank": 1},
        {"target_id": "b2", "stratum": "b", "selection_rank": 2},
    ]
    ordered = _ranked_throughput_order(records, strata)
    assert [record["target_id"] for record in ordered[:10]] == ["a0", "b0", "b1", "a1", "b2"]


def test_existing_mismatched_output_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "benchmark"
    output.mkdir()
    (output / "dataset.json").write_text('{"dataset_id": "different"}', encoding="utf-8")

    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", _fake_request)
    with pytest.raises(ValueError, match="benchmark output already exists"):
        prepare_benchmark_dataset(output, _make_spec(), workers=2, progress=lambda _message: None)


def test_non_matching_candidate_is_rejected_by_candidate_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", _fake_request)
    spec = _make_spec()

    _identifier, record, monomer_rejection = curator._candidate_outcome("3BAD-1", spec, spec.strata[0])
    assert record is None
    assert monomer_rejection == "reference_length_does_not_match_the_stratum"

    _identifier, record, dimer_rejection = curator._candidate_outcome("3BAD-1", spec, spec.strata[1])
    assert record is None
    assert dimer_rejection == "reference_chain_count_does_not_match_the_stratum"


def test_gunzip_bounded_passes_normal_payload() -> None:
    payload = PDB_TEXTS["1abc"].encode()
    assert _gunzip_bounded(gzip.compress(payload)) == payload


def test_gunzip_bounded_rejects_expansion_beyond_limit() -> None:
    payload = b"A" * 1024
    with pytest.raises(CandidateRejected, match="exceeds size limit"):
        _gunzip_bounded(gzip.compress(payload), maximum_bytes=512)


def test_gunzip_bounded_rejects_invalid_gzip() -> None:
    with pytest.raises(OSError):
        _gunzip_bounded(b"not a gzip payload")


def test_candidate_rejected_is_a_value_error() -> None:
    assert issubclass(CandidateRejected, ValueError)
    with pytest.raises(ValueError):
        raise CandidateRejected("reference length does not match the stratum")


def test_candidate_outcome_rejects_decompression_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    bomb = gzip.compress(b"A" * (MAX_DECOMPRESSED_BYTES + 1))

    def bomb_request(
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout: int = 60,
        maximum_bytes: int | None = None,
    ) -> bytes:
        if url.startswith(f"{DATA_API}/assembly/"):
            return json.dumps(ASSEMBLIES["1abc"]).encode()
        if url.startswith(f"{DATA_API}/entry/"):
            return json.dumps(ENTRIES["1abc"]).encode()
        if url.startswith(f"{FILES_URL}/"):
            return bomb
        raise AssertionError(f"unexpected request URL: {url}")

    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", bomb_request)
    spec = _make_spec()

    _identifier, record, rejection = curator._candidate_outcome("1ABC-1", spec, spec.strata[0])
    assert record is None
    assert rejection == "decompressed_coordinate_file_exceeds_size_limit"


def test_dataset_preserves_raw_specification_and_description(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "benchmark"
    _run_curation(tmp_path, monkeypatch, output)

    dataset_json = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    spec = _make_spec()
    assert dataset_json["description"] == spec.description
    assert dataset_json["specification"] == dict(spec.raw_specification)


def test_synthetic_record_fingerprint_matches_upstream_formula() -> None:
    spec = _make_spec()
    records = [
        (
            "pdb_1abc_assembly_1",
            "a" * 64,
            "monomer_short",
            "b" * 64,
            "c" * 64,
        ),
    ]

    digest = hashlib.sha256()
    digest.update(b"afdb-pdb-benchmark-v1\0")
    digest.update(json.dumps(dict(spec.raw_specification), sort_keys=True, separators=(",", ":")).encode())
    for target_id, sequence_sha256, stratum, reference_sha256, source_gzip_sha256 in records:
        value = {
            "target_id": target_id,
            "sequence_sha256": sequence_sha256,
            "stratum": stratum,
            "reference_sha256": reference_sha256,
            "source_gzip_sha256": source_gzip_sha256,
        }
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())

    assert _dataset_fingerprint(dict(spec.raw_specification), records) == digest.hexdigest()


def test_changing_top_level_description_changes_fingerprint() -> None:
    spec = _make_spec()
    records = [("t1", "a" * 64, "monomer_short", "b" * 64, "c" * 64)]
    base = _dataset_fingerprint(dict(spec.raw_specification), records)

    changed = dict(spec.raw_specification)
    changed["description"] = "a different description"

    assert _dataset_fingerprint(changed, records) != base


def test_curation_exhaustion_raises_benchmark_curation_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", _fake_request)
    spec = replace(
        _make_spec(),
        strata=(
            BenchmarkStratum(
                name="monomer_short",
                chain_count=1,
                minimum_total_residues=1,
                maximum_total_residues=2,
                count=2,
            ),
        ),
    )
    output = tmp_path / "benchmark"

    with pytest.raises(BenchmarkCurationFailed, match="staging="):
        prepare_benchmark_dataset(output, spec, workers=2, progress=lambda _message: None)


def test_dataset_fingerprint_mmcif_prefix_differs() -> None:
    """Verify mmCIF-profile fingerprint differs from gzip-profile."""
    spec = _make_spec()
    records = [("t1", "a" * 64, "monomer_short", "b" * 64, "c" * 64)]
    gzip_fp = _dataset_fingerprint(dict(spec.raw_specification), records)
    mmcif_fp = _dataset_fingerprint(
        dict(spec.raw_specification),
        records,
        source_key="source_mmcif_sha256",
        fingerprint_prefix=b"afdb-pdb-benchmark-mmcif-v1\0",
    )
    assert gzip_fp != mmcif_fp


def test_dataset_fingerprint_source_key_threads_through() -> None:
    """Verify per-record JSON uses correct key name."""
    spec = _make_spec()
    records = [("t1", "a" * 64, "monomer_short", "b" * 64, "c" * 64)]
    mmcif_fp = _dataset_fingerprint(
        dict(spec.raw_specification),
        records,
        source_key="source_mmcif_sha256",
        fingerprint_prefix=b"afdb-pdb-benchmark-mmcif-v1\0",
    )
    # Manually compute with the mmCIF key.
    digest = hashlib.sha256()
    digest.update(b"afdb-pdb-benchmark-mmcif-v1\0")
    digest.update(json.dumps(dict(spec.raw_specification), sort_keys=True, separators=(",", ":")).encode())
    value = {
        "target_id": "t1",
        "sequence_sha256": "a" * 64,
        "stratum": "monomer_short",
        "reference_sha256": "b" * 64,
        "source_mmcif_sha256": "c" * 64,
    }
    digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    assert mmcif_fp == digest.hexdigest()


def test_materialize_dataset_selection_description_includes_rejections(
    tmp_path: Path,
) -> None:
    """Verify that when selection_description is passed, the output dataset.json selection
    block includes the rejections key."""
    from bspp.orchestration.control.folding_benchmark.curator import _materialize_dataset

    strata = (
        BenchmarkStratum(
            name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
        ),
    )
    source = {"provider": "RCSB PDB"}
    filters = {"minimum_coordinate_coverage": 0.7}
    raw = {
        "schema_version": 1,
        "dataset_id": "test-benchmark",
        "description": "test",
        "selection_seed": "x",
        "source": source,
        "filters": filters,
        "strata": [
            {
                "name": "monomer_short",
                "chain_count": 1,
                "minimum_total_residues": 1,
                "maximum_total_residues": 100,
                "count": 1,
            }
        ],
        "throughput_subset_sizes": [10],
    }
    spec = BenchmarkSpec(
        schema_version=1,
        dataset_id="test-benchmark",
        selection_seed="x",
        source=source,
        filters=filters,
        strata=strata,
        throughput_subset_sizes=(10,),
        description="test",
        raw_specification=raw,
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    # Minimal record
    record = {
        "target_id": "pdb_1abc_assembly_1",
        "description": "test",
        "pdb_id": "1ABC",
        "assembly_id": "1",
        "stratum": "monomer_short",
        "chain_count": 1,
        "chain_ids": ["A"],
        "chain_lengths": [2],
        "chains": ["AG"],
        "sequence": "AG",
        "sequence_sha256": "x" * 64,
        "total_residues": 2,
        "ca_counts": [2],
        "coordinate_coverage": [1.0],
        "minimum_coordinate_coverage": 1.0,
        "cluster_identity": 30,
        "cluster_ids": ["100"],
        "cluster_signature": "100",
        "release_date": "2023-05-15",
        "experimental_methods": ["X-RAY DIFFRACTION"],
        "resolution_angstrom": 2.0,
        "source_url": "https://example.com",
        "source_gzip_sha256": "y" * 64,
        "reference_sha256": "z" * 64,
        "reference_bytes": b"END\n",
        "selection_rank": 0,
    }
    selection_desc = {
        "monomers": "pinned target list (reconstruction)",
        "complexes": "pinned target list (reconstruction)",
        "candidate_order": "pinned target_id sort order",
        "rejections": {},
    }
    dataset = _materialize_dataset(
        staging,
        spec,
        [record],
        {},
        selection_description=selection_desc,
    )
    assert dataset["selection"]["rejections"] == {}
    assert dataset["selection"]["monomers"] == "pinned target list (reconstruction)"


def test_materialize_dataset_no_candidates_param() -> None:
    """Verify _materialize_dataset no longer accepts a candidates parameter."""
    import inspect

    from bspp.orchestration.control.folding_benchmark.curator import _materialize_dataset

    sig = inspect.signature(_materialize_dataset)
    assert "candidates" not in sig.parameters
