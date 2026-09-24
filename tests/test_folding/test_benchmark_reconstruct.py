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

"""Tests for the reconstruction path (pinned target list → mmCIF → re-pinned corpus)."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from bspp.orchestration.control.folding_benchmark.curator import (
    DATA_API,
    FILES_URL,
    MAX_DOWNLOAD_BYTES,
    BenchmarkCurationFailed,
    _dataset_fingerprint,
    reconstruct_benchmark_dataset,
)
from bspp.orchestration.control.folding_benchmark.spec import BenchmarkSpec, BenchmarkStratum

# --- mmCIF fixture helpers ---

_MMCIF_HEADER = """\
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
"""


def _mmcif_row(
    group: str = "ATOM",
    atom_id: str = "CA",
    alt_id: str = ".",
    comp_id: str = "ALA",
    asym_id: str = "A",
    seq_id: str = "1",
    ins_code: str = "?",
    x: str = "10.000",
    y: str = "20.000",
    z: str = "30.000",
) -> str:
    return f"{group} 1 C {atom_id} {alt_id} {comp_id} {asym_id} 1 {seq_id} {ins_code} {x} {y} {z} 1.00 0.00"


def _mmcif_text(*rows: str) -> str:
    return _MMCIF_HEADER + "\n".join(rows) + "\n"


# --- Spec helper ---


def _make_spec(
    strata: tuple[BenchmarkStratum, ...] | None = None,
) -> BenchmarkSpec:
    if strata is None:
        strata = (
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
            BenchmarkStratum(
                name="dimer_small", chain_count=2, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    source = {"provider": "RCSB PDB"}
    filters = {"minimum_coordinate_coverage": 0.7}
    raw = {
        "schema_version": 1,
        "dataset_id": "pdb-temporal-2022-2025-v1",
        "description": "test-benchmark",
        "selection_seed": "test-seed",
        "source": source,
        "filters": filters,
        "strata": [
            {
                "name": s.name,
                "chain_count": s.chain_count,
                "minimum_total_residues": s.minimum_total_residues,
                "maximum_total_residues": s.maximum_total_residues,
                "count": s.count,
            }
            for s in strata
        ],
        "throughput_subset_sizes": [10],
    }
    return BenchmarkSpec(
        schema_version=1,
        dataset_id="pdb-temporal-2022-2025-v1",
        selection_seed="test-seed",
        source=source,
        filters=filters,
        strata=strata,
        throughput_subset_sizes=(10,),
        description="test-benchmark",
        raw_specification=raw,
    )


# --- Target list helpers ---


def _target_record(
    target_id: str,
    chains: list[str],
    chain_lengths: list[int],
    total_length: int,
    sequence_sha256: str,
    description: str | None = None,
) -> dict[str, Any]:
    if description is None:
        # Parse pdb_id and assembly_id from target_id
        import re

        m = re.match(r"^pdb_([a-z0-9]+)_assembly_([0-9]+)$", target_id)
        assert m is not None
        pdb_id = m.group(1).upper()
        assembly_id = m.group(2)
        description = (
            f"{target_id} RCSB PDB {pdb_id} biological assembly {assembly_id}; "
            f"stratum=monomer_short; release=2023-05-15"
        )
    return {
        "target_id": target_id,
        "description": description,
        "chains": chains,
        "chain_lengths": chain_lengths,
        "total_length": total_length,
        "sequence_sha256": sequence_sha256,
    }


def _seq_sha(chains: list[str]) -> str:
    return hashlib.sha256(":".join(chains).encode()).hexdigest()


def _write_targets(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# --- Mock _request ---


def _make_entry_json(method: str = "X-RAY DIFFRACTION", resolution: float = 2.0) -> dict[str, Any]:
    return {
        "rcsb_accession_info": {"initial_release_date": "2023-05-15"},
        "exptl": [{"method": method}],
        "rcsb_entry_info": {"resolution_combined": [resolution]},
    }


def _make_fake_request(
    mmcif_texts: dict[str, str],
    entry_jsons: dict[str, dict[str, Any]] | None = None,
    captured: dict[str, Any] | None = None,
):
    if entry_jsons is None:
        entry_jsons = {}
    if captured is None:
        captured = {}

    def _fake_request(
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout: int = 60,
        maximum_bytes: int | None = None,
    ) -> bytes:
        if url.startswith(f"{DATA_API}/entry/"):
            pdb_id_upper = url.split("/")[-1]
            pdb_id = pdb_id_upper.lower()
            return json.dumps(entry_jsons.get(pdb_id, _make_entry_json())).encode()
        if url.startswith(f"{FILES_URL}/"):
            # Extract pdb_id from URL: <pdb_id>-assembly<n>.cif.gz
            filename = url.split("/")[-1]
            pdb_id = filename.split("-assembly")[0].lower()
            if captured is not None:
                captured["mmcif_maximum_bytes"] = maximum_bytes
                captured["mmcif_url"] = url
            return gzip.compress(mmcif_texts[pdb_id].encode())
        raise AssertionError(f"unexpected request URL: {url}")

    return _fake_request


# --- Tests ---


def test_reconstruct_from_target_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Small pinned targets.jsonl (2 records incl. one multi-chain), mock reconstruction."""
    # Monomer: ALA GLY (2 residues, 1 chain)
    monomer_chains = ["AG"]
    monomer_sha = _seq_sha(monomer_chains)
    monomer_mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    # Dimer: chain A = ALA, chain B = GLY
    dimer_chains = ["A", "G"]
    dimer_sha = _seq_sha(dimer_chains)
    dimer_mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="B", seq_id="1"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            monomer_chains,
            [2],
            2,
            monomer_sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        ),
        _target_record(
            "pdb_2def_assembly_1",
            dimer_chains,
            [1, 1],
            2,
            dimer_sha,
            description="pdb_2def_assembly_1 RCSB PDB 2DEF biological assembly 1; "
            "stratum=dimer_small; release=2023-05-15",
        ),
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec()
    mmcif_texts = {"1abc": monomer_mmcif, "2def": dimer_mmcif}
    fake = _make_fake_request(mmcif_texts)
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    dataset = reconstruct_benchmark_dataset(output, spec, target_list, workers=2, progress=lambda _: None)
    assert dataset["targets"] == 2
    assert (output / "dataset.json").is_file()
    assert (output / "targets.jsonl").is_file()
    assert (output / "references" / "pdb_1abc_assembly_1.pdb").is_file()
    assert (output / "references" / "pdb_2def_assembly_1.pdb").is_file()


def test_reconstruct_sequence_sha256_matches_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert public[0]["sequence_sha256"] == sha


def test_reconstruct_pinned_chains_as_seqres(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    ref_text = (output / "references" / "pdb_1abc_assembly_1.pdb").read_text()
    assert "SEQRES" in ref_text
    assert "ALA" in ref_text and "GLY" in ref_text


def test_reconstruct_partial_coverage_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A target where CA records cover only ~80% of the pinned sequence is accepted."""
    chains = ["AGSGS"]  # 5 residues
    sha = _seq_sha(chains)
    # Only 4 out of 5 CA atoms (80% coverage)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
        _mmcif_row(comp_id="SER", asym_id="A", seq_id="3"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="4"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [5],
            5,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert public[0]["coordinate_coverage"][0] == 0.8


def test_reconstruct_heteromer_chain_matching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2-chain heteromer where label_asym_id order differs from pinned chains order."""
    # Pinned: chain 0 = "A" (ALA), chain 1 = "G" (GLY)
    chains = ["A", "G"]
    sha = _seq_sha(chains)
    # mmCIF has asym_id "B" = ALA, asym_id "A" = GLY (reversed order)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="ALA", asym_id="B", seq_id="1"),
    )
    targets = [
        _target_record(
            "pdb_2def_assembly_1",
            chains,
            [1, 1],
            2,
            sha,
            description="pdb_2def_assembly_1 RCSB PDB 2DEF biological assembly 1; "
            "stratum=dimer_small; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="dimer_small", chain_count=2, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"2def": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert public[0]["chain_ids"] == ["A", "B"]


def test_reconstruct_homomer_chain_matching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N9: A 2-chain homomer (identical sequences) — assert concrete assignment."""
    chains = ["A", "A"]  # Both ALA
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="ALA", asym_id="B", seq_id="1"),
    )
    targets = [
        _target_record(
            "pdb_2def_assembly_1",
            chains,
            [1, 1],
            2,
            sha,
            description="pdb_2def_assembly_1 RCSB PDB 2DEF biological assembly 1; "
            "stratum=dimer_small; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="dimer_small", chain_count=2, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"2def": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert public[0]["chain_ids"] == ["A", "B"]
    assert public[0]["chain_count"] == 2


def test_reconstruct_augmenting_path_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: greedy matching would falsely reject; Kuhn's finds the perfect matching."""
    chains = ["AG", "G"]  # (ALA, GLY) and (GLY,)
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="ALA", asym_id="B", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="B", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_2def_assembly_1",
            chains,
            [2, 1],
            3,
            sha,
            description="pdb_2def_assembly_1 RCSB PDB 2DEF biological assembly 1; "
            "stratum=dimer_small; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="dimer_small", chain_count=2, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"2def": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert public[0]["chain_ids"] == ["A", "B"]


def test_reconstruct_ambiguous_matching_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A target where no perfect bipartite matching exists is rejected."""
    chains = ["A", "G"]  # ALA and GLY
    sha = _seq_sha(chains)
    # mmCIF has both chains as ALA — neither matches GLY
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="ALA", asym_id="B", seq_id="1"),
    )
    targets = [
        _target_record(
            "pdb_2def_assembly_1",
            chains,
            [1, 1],
            2,
            sha,
            description="pdb_2def_assembly_1 RCSB PDB 2DEF biological assembly 1; "
            "stratum=dimer_small; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="dimer_small", chain_count=2, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"2def": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    with pytest.raises(BenchmarkCurationFailed):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)


def test_reconstruct_multi_char_label_asym_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B1: Multi-character label_asym_id values → synthetic single-character IDs."""
    chains = ["A", "G"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="AA", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="BB", seq_id="1"),
    )
    targets = [
        _target_record(
            "pdb_2def_assembly_1",
            chains,
            [1, 1],
            2,
            sha,
            description="pdb_2def_assembly_1 RCSB PDB 2DEF biological assembly 1; "
            "stratum=dimer_small; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="dimer_small", chain_count=2, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"2def": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert public[0]["chain_ids"] == ["A", "B"]
    # validation-suite.json chain_map should be identity
    vs = json.loads((output / "validation-suite.json").read_text())
    case = vs["cases"][0]
    assert case["chain_map"] == {"A": "A", "B": "B"}


def test_reconstruct_fingerprint_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run reconstruction twice with same mocks; verify identical fingerprints."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    for suffix in ("first", "second"):
        tl = tmp_path / f"targets_{suffix}.jsonl"
        _write_targets(tl, targets)
        out = tmp_path / f"corpus_{suffix}"
        monkeypatch.setattr(curator, "_request", fake)
        reconstruct_benchmark_dataset(out, spec, tl, workers=1, progress=lambda _: None)
    fp1 = json.loads((tmp_path / "corpus_first" / "dataset.json").read_text())["dataset_fingerprint"]
    fp2 = json.loads((tmp_path / "corpus_second" / "dataset.json").read_text())["dataset_fingerprint"]
    assert fp1 == fp2


def test_reconstruct_fingerprint_uses_mmcif_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fingerprint uses b\"afdb-pdb-benchmark-mmcif-v1\\0\" prefix and source_mmcif_sha256 key."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    expected_fp = _dataset_fingerprint(
        dict(spec.raw_specification),
        [
            (r["target_id"], r["sequence_sha256"], r["stratum"], r["reference_sha256"], r["source_mmcif_sha256"])
            for r in public
        ],
        source_key="source_mmcif_sha256",
        fingerprint_prefix=b"afdb-pdb-benchmark-mmcif-v1\0",
    )
    actual_fp = json.loads((output / "dataset.json").read_text())["dataset_fingerprint"]
    assert actual_fp == expected_fp


def test_reconstruct_unrecoverable_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    r = public[0]
    assert r["cluster_ids"] == []
    assert r["cluster_identity"] is None
    assert r["cluster_signature"] == ""
    assert "selection_rank" in r
    assert "ca_counts" in r
    assert "coordinate_coverage" in r
    assert "assembly_composition" not in r
    assert "assembly_details" not in r
    assert "assembly_oligomeric_details" not in r
    assert "source_gzip_sha256" not in r
    assert "source_pdb_sha256" not in r


def test_reconstruct_reference_bytes_present_and_stripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N4: reference_bytes is present in internal records but absent from targets.jsonl."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert "reference_bytes" not in public[0]


def test_reconstruct_metadata_from_data_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    entry = _make_entry_json(method="ELECTRON MICROSCOPY", resolution=3.5)
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif}, entry_jsons={"1abc": entry})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert "ELECTRON MICROSCOPY" in public[0]["experimental_methods"]
    assert public[0]["resolution_angstrom"] == 3.5


def test_reconstruct_description_prefix_stripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    assert not public[0]["description"].startswith("pdb_1abc_assembly_1 ")


def test_reconstruct_output_exists_guard_raises(tmp_path: Path) -> None:
    output = tmp_path / "corpus"
    output.mkdir()
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, [])
    spec = _make_spec()
    with pytest.raises(ValueError, match="reconstruction output already exists"):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)


def test_reconstruct_output_exists_guard_legacy_corpus_raises(tmp_path: Path) -> None:
    output = tmp_path / "corpus"
    output.mkdir()
    spec = _make_spec()
    (output / "dataset.json").write_text(
        json.dumps(
            {
                "dataset_id": spec.dataset_id,
                "specification": dict(spec.raw_specification),
            }
        ),
        encoding="utf-8",
    )
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, [])
    with pytest.raises(ValueError, match="reconstruction output already exists"):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)


def test_reconstruct_atomic_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Output is published via tempfile.mkdtemp + os.replace."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    assert output.is_dir()
    # No staging directories should remain
    siblings = [p for p in output.parent.iterdir() if p.name.startswith(".")]
    assert all(not p.name.endswith(".staging") for p in siblings)


def test_reconstruct_per_stratum_count_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If a stratum's count is wrong, BenchmarkCurationFailed is raised."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    # Spec expects 2 monomer_short targets but we only provide 1
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=2
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    with pytest.raises(BenchmarkCurationFailed):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)


def test_reconstruct_source_mmcif_sha256_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N10: source_mmcif_sha256 == SHA-256 of raw decompressed bytes."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    decompressed = mmcif.encode()
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    public = [json.loads(line) for line in (output / "targets.jsonl").read_text().splitlines()]
    expected = hashlib.sha256(decompressed).hexdigest()
    assert public[0]["source_mmcif_sha256"] == expected


def test_reconstruct_workers_validation(tmp_path: Path) -> None:
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, [])
    spec = _make_spec()
    output = tmp_path / "corpus"
    with pytest.raises(ValueError, match="workers must be an integer"):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=0, progress=lambda _: None)
    with pytest.raises(ValueError, match="workers must be an integer"):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=True, progress=lambda _: None)  # type: ignore[arg-type]


def test_reconstruct_selection_has_rejections_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    dataset_json = json.loads((output / "dataset.json").read_text())
    assert "rejections" in dataset_json["selection"]
    assert dataset_json["selection"]["rejections"] == {}


def test_reconstruct_summary_includes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    dataset = reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    assert "output" in dataset
    assert dataset["output"] == str(output.resolve())


def test_reconstruct_output_parent_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N6: A nested --output path whose parent does not exist is created successfully."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "nested" / "deep" / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    assert output.is_dir()


def test_reconstruct_chain_count_cross_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N8: A target whose pinned chains count differs from stratum.chain_count is rejected."""
    # Pinned has 1 chain but stratum expects 2
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=dimer_small; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="dimer_small", chain_count=2, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    with pytest.raises(BenchmarkCurationFailed):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)


def test_reconstruct_download_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N5: Mock _request to verify maximum_bytes=MAX_DOWNLOAD_BYTES is passed for mmCIF download."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="ALA", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="GLY", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    captured: dict[str, Any] = {}
    fake = _make_fake_request({"1abc": mmcif}, captured=captured)
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
    assert captured.get("mmcif_maximum_bytes") == MAX_DOWNLOAD_BYTES


def test_reconstruct_per_target_rejection_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """N3: A rejected target's target_id and reason appear in progress output."""
    chains = ["AG"]
    sha = _seq_sha(chains)
    # mmCIF has no matching CA residues (wrong residue types)
    mmcif = _mmcif_text(
        _mmcif_row(comp_id="VAL", asym_id="A", seq_id="1"),
        _mmcif_row(comp_id="LEU", asym_id="A", seq_id="2"),
    )
    targets = [
        _target_record(
            "pdb_1abc_assembly_1",
            chains,
            [2],
            2,
            sha,
            description="pdb_1abc_assembly_1 RCSB PDB 1ABC biological assembly 1; "
            "stratum=monomer_short; release=2023-05-15",
        )
    ]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec(
        strata=(
            BenchmarkStratum(
                name="monomer_short", chain_count=1, minimum_total_residues=1, maximum_total_residues=100, count=1
            ),
        )
    )
    fake = _make_fake_request({"1abc": mmcif})
    import bspp.orchestration.control.folding_benchmark.curator as curator

    monkeypatch.setattr(curator, "_request", fake)
    messages: list[str] = []
    with pytest.raises(BenchmarkCurationFailed):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda msg: messages.append(msg))
    assert any("pdb_1abc_assembly_1" in m for m in messages)


def test_reconstruct_duplicate_target_id_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1: duplicate target_id values are rejected before reconstruction."""
    chains = ["A"]
    sha = _seq_sha(chains)
    record = _target_record("pdb_1abc_assembly_1", chains, [1], 1, sha)
    targets = [record, dict(record)]
    target_list = tmp_path / "targets.jsonl"
    _write_targets(target_list, targets)
    output = tmp_path / "corpus"
    spec = _make_spec()
    with pytest.raises(ValueError, match="duplicate target_id"):
        reconstruct_benchmark_dataset(output, spec, target_list, workers=1, progress=lambda _: None)
