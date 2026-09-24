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

"""Cross-story folding benchmark integration test (curator → runtime → control).

This pins the vertical seam that previously existed only as a throwaway probe:
a small JSON ``BenchmarkSpec`` is curated with only the RCSB HTTP boundary
mocked, the curator output is fetched through a fake ``s3_transfer.cp`` with the
recursive prefix normalization and the real checksum/fingerprint verification,
a synthetic completed run is validated through the real ``validate_run``, and
the control-side summary acceptance gate consumes the exact runtime-produced
summary.
"""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from bspp.orchestration.control.folding_benchmark.curator import (
    DATA_API,
    FILES_URL,
    SEARCH_URL,
    prepare_benchmark_dataset,
)
from bspp.orchestration.control.folding_benchmark.spec import load_benchmark_spec
from bspp.orchestration.control.folding_benchmark_submit import _summary_accepted
from bspp.orchestration.runtime.data_movement.common import TransferResult
from bspp.orchestration.runtime.folding.benchmark.corpus import fetch_pinned_corpus
from bspp.orchestration.runtime.folding.benchmark.index import (
    build_canonical_pair_index,
    write_canonical_pair_index,
)
from bspp.orchestration.runtime.folding.benchmark.suite import load_validation_suite
from bspp.orchestration.runtime.folding.benchmark.validation import validate_run


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


def _write_spec(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "integration-benchmark",
                "description": "integration test benchmark",
                "selection_seed": "integration-seed",
                "source": {
                    "provider": "RCSB PDB",
                    "release_date_min": "2022-01-01",
                    "release_date_max": "2025-12-31",
                    "max_resolution_angstrom": 3.0,
                    "required_any_experimental_methods": ["X-RAY DIFFRACTION"],
                    "assembly_id": "1",
                },
                "filters": {
                    "minimum_chain_length": 1,
                    "minimum_coordinate_coverage": 0.5,
                    "canonical_amino_acids_only": True,
                    "require_candidate_biological_assembly": True,
                    "monomer_sequence_cluster_identity": 30,
                },
                "strata": [
                    {
                        "name": "monomer_short",
                        "chain_count": 1,
                        "minimum_total_residues": 1,
                        "maximum_total_residues": 2,
                        "count": 1,
                    }
                ],
                "throughput_subset_sizes": [10],
            }
        ),
        encoding="utf-8",
    )
    return path


def _assembly() -> dict[str, Any]:
    return {
        "pdbx_struct_assembly": {
            "rcsb_candidate_assembly": "Y",
            "details": "author_and_software_defined_assembly",
        },
        "rcsb_assembly_info": {"polymer_composition": "Protein"},
    }


def _entry() -> dict[str, Any]:
    return {
        "rcsb_accession_info": {"initial_release_date": "2023-05-15"},
        "exptl": [{"method": "X-RAY DIFFRACTION"}],
        "rcsb_entry_info": {"resolution_combined": [2.0]},
        "rcsb_entry_container_identifiers": {"polymer_entity_ids": ["1"]},
    }


def _entity() -> dict[str, Any]:
    return {
        "entity_poly": {"pdbx_seq_one_letter_code_can": "AG"},
        "rcsb_cluster_membership": [{"identity": 30, "cluster_id": 100}],
    }


def _fake_request(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: int = 60,
    maximum_bytes: int | None = None,
) -> bytes:
    if url == SEARCH_URL:
        return json.dumps({"result_set": ["1abc-1"]}).encode()
    if url.startswith(f"{DATA_API}/assembly/"):
        return json.dumps(_assembly()).encode()
    if url.startswith(f"{DATA_API}/entry/"):
        return json.dumps(_entry()).encode()
    if url.startswith(f"{DATA_API}/polymer_entity/"):
        return json.dumps(_entity()).encode()
    if url.startswith(f"{FILES_URL}/"):
        return gzip.compress(_pdb_text({"A": ("ALA", "GLY")}).encode())
    raise AssertionError(f"unexpected request URL: {url}")


def _install_fake_transfer(
    monkeypatch: pytest.MonkeyPatch,
    source_dir: Path,
) -> dict[str, object]:
    calls: dict[str, object] = {"src": None, "credentials": None}

    def _cp(
        src: str | Path,
        dst: str | Path,
        *,
        credentials: object = None,
        numworkers: int | None = None,
        extra_args: tuple[str, ...] = (),
        dry_run: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> TransferResult:
        calls["src"] = str(src)
        calls["credentials"] = credentials
        shutil.copytree(source_dir, Path(dst), dirs_exist_ok=True)
        return TransferResult(tool="s5cmd", argv=("cp", str(src), str(dst)), returncode=0, elapsed_s=0.0)

    monkeypatch.setattr("bspp.orchestration.runtime.data_movement.s3.transfer.cp", _cp)
    return calls


def test_full_vertical_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bspp.orchestration.control.folding_benchmark.curator as curator

    spec_path = _write_spec(tmp_path / "spec.json")
    spec = load_benchmark_spec(spec_path)
    assert spec.description == "integration test benchmark"

    monkeypatch.setattr(curator, "_request", _fake_request)
    curated_dir = tmp_path / "curated"
    dataset = prepare_benchmark_dataset(curated_dir, spec, workers=2, progress=lambda _message: None)

    # Curator output preserves the verbatim specification and description.
    dataset_json = json.loads((curated_dir / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json["description"] == spec.description
    assert dataset_json["specification"] == dict(spec.raw_specification)
    assert dataset_json["dataset_fingerprint"] == dataset["dataset_fingerprint"]

    # The curated validation suite loads through the strict public loader.
    suite = load_validation_suite(curated_dir / "validation-suite.json")
    assert suite.dataset_id == "integration-benchmark"
    assert suite.fingerprint == dataset_json["dataset_fingerprint"]
    assert len(suite.cases) == 1

    # Fetch the curator output through the real corpus boundary.
    calls = _install_fake_transfer(monkeypatch, curated_dir)
    corpus_dir = tmp_path / "corpus"
    fetch_pinned_corpus(
        "s3://benchmarks/integration-benchmark/",
        corpus_dir,
        expected_fingerprint=dataset_json["dataset_fingerprint"],
    )
    assert calls["src"] == "s3://benchmarks/integration-benchmark/*"

    records = [json.loads(line) for line in (corpus_dir / "targets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    target_id = record["target_id"]
    sequence_sha256 = record["sequence_sha256"]

    # Build and write the canonical-pair index through the public index APIs.
    index_path = tmp_path / "index.json"
    index = build_canonical_pair_index(
        run_id="integration-run",
        entries=[
            (
                target_id,
                sequence_sha256,
                "AF-0000000000000001",
                "ColabFold v1.6.0 / AlphaFold-Multimer",
                "AF-0000000000000001-model_v1.pdb",
                "AF-0000000000000001-meta_v1.json",
            )
        ],
    )
    write_canonical_pair_index(index, index_path)

    # Materialize a synthetic completed run whose prediction matches the reference.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    reference_path = corpus_dir / str(record["reference_path"])
    (run_dir / "AF-0000000000000001-model_v1.pdb").write_bytes(reference_path.read_bytes())
    (run_dir / "AF-0000000000000001-meta_v1.json").write_text(
        json.dumps({"schema_version": 1, "plddt": [90.0], "pae": [[1.0]], "max_pae": 1.0}),
        encoding="utf-8",
    )

    evidence_dir = tmp_path / "evidence"
    summary = validate_run(
        run_dir,
        curated_dir / "validation-suite.json",
        index_path,
        corpus_dir,
        output_dir=evidence_dir,
    )

    assert summary["case_count"] == 1
    assert summary["passed_count"] == 1
    case = summary["cases"][0]
    assert case["identity_valid"] is True
    assert case["quality_valid"] is True
    assert case["structure_valid"] is True
    assert case["passed"] is True

    assert (evidence_dir / "validation.parquet").is_file()
    assert (evidence_dir / "summary.json").is_file()
    assert _summary_accepted(summary) is True
