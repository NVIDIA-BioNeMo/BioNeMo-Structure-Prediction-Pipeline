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

"""Tests for the C-alpha parser, Kabsch alignment, and structure check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bspp.orchestration.runtime.folding.benchmark.validation import (
    ca_metrics,
    parse_pdb_ca,
    validate_run,
)


def _atom_line(
    serial: int,
    resseq: int,
    x: float,
    y: float,
    z: float,
    *,
    chain: str = "A",
    atom_name: str = " CA ",
    altloc: str = " ",
    resname: str = "ALA",
    icode: str = " ",
) -> str:
    """Build one fixed-column PDB ``ATOM`` record."""
    return (
        f"ATOM  {serial:>5} {atom_name}{altloc}{resname} {chain}{resseq:>4}{icode}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
    )


def _write_pdb(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines), encoding="utf-8")


def test_parse_pdb_ca_basic(tmp_path: Path) -> None:
    path = tmp_path / "basic.pdb"
    _write_pdb(
        path,
        [
            _atom_line(1, 1, 11.104, 13.207, 10.000),
            _atom_line(2, 2, 12.000, 14.000, 11.000),
        ],
    )
    chains = parse_pdb_ca(path)
    assert set(chains) == {"A"}
    assert chains["A"] == [(11.104, 13.207, 10.000), (12.000, 14.000, 11.000)]


def test_parse_pdb_ca_blank_chain_is_underscore(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdb"
    _write_pdb(path, [_atom_line(1, 1, 1.0, 2.0, 3.0, chain=" ")])
    chains = parse_pdb_ca(path)
    assert set(chains) == {"_"}
    assert chains["_"] == [(1.0, 2.0, 3.0)]


def test_parse_pdb_ca_accepts_altloc_a(tmp_path: Path) -> None:
    path = tmp_path / "altloc.pdb"
    _write_pdb(path, [_atom_line(1, 1, 1.0, 2.0, 3.0, altloc="A")])
    chains = parse_pdb_ca(path)
    assert chains["A"] == [(1.0, 2.0, 3.0)]


def test_parse_pdb_ca_skips_non_ca_and_non_atom(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdb"
    _write_pdb(
        path,
        [
            "HEADER    synthetic\n",
            _atom_line(1, 1, 1.0, 2.0, 3.0, atom_name=" N  "),
            "HETATM    2  CA  ALA A   1       4.000   5.000   6.000  1.00 20.00           C\n",
            "TER\n",
            _atom_line(3, 2, 7.0, 8.0, 9.0),
        ],
    )
    chains = parse_pdb_ca(path)
    assert chains["A"] == [(7.0, 8.0, 9.0)]


def test_parse_pdb_ca_first_model_only(tmp_path: Path) -> None:
    path = tmp_path / "models.pdb"
    _write_pdb(
        path,
        [
            "MODEL        1\n",
            _atom_line(1, 1, 1.0, 2.0, 3.0),
            _atom_line(2, 2, 4.0, 5.0, 6.0),
            "ENDMDL\n",
            "MODEL        2\n",
            _atom_line(3, 1, 7.0, 8.0, 9.0),
            "ENDMDL\n",
        ],
    )
    chains = parse_pdb_ca(path)
    assert chains["A"] == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


def test_parse_pdb_ca_deduplicates_residues(tmp_path: Path) -> None:
    path = tmp_path / "dedup.pdb"
    _write_pdb(
        path,
        [
            _atom_line(1, 1, 1.0, 2.0, 3.0),
            _atom_line(2, 1, 9.0, 9.0, 9.0),
        ],
    )
    chains = parse_pdb_ca(path)
    assert chains["A"] == [(1.0, 2.0, 3.0)]


def test_parse_pdb_ca_skips_malformed_coordinate(tmp_path: Path) -> None:
    path = tmp_path / "malformed.pdb"
    valid = _atom_line(1, 1, 1.0, 2.0, 3.0)
    malformed = valid[:30] + "bad     " + valid[38:]
    _write_pdb(path, [malformed, _atom_line(2, 2, 4.0, 5.0, 6.0)])
    chains = parse_pdb_ca(path)
    assert chains["A"] == [(4.0, 5.0, 6.0)]


def test_parse_pdb_ca_empty_result(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdb"
    _write_pdb(path, ["HEADER    no atoms\n", "END\n"])
    assert parse_pdb_ca(path) == {}


def test_ca_metrics_identical() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    metrics = ca_metrics(coordinates, coordinates.copy())
    assert metrics["ca_rmsd"] == 0.0
    assert metrics["ca_coverage"] == 1.0
    assert metrics["matched_ca_atoms"] == 3
    assert metrics["predicted_ca_atoms"] == 3
    assert metrics["reference_ca_atoms"] == 3
    assert metrics["ca_match_mode"] == "full"


def test_ca_metrics_rigid_transform() -> None:
    rng = np.random.default_rng(0)
    coordinates = rng.standard_normal((5, 3))
    rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    transformed = coordinates @ rotation.T + rng.standard_normal(3)
    metrics = ca_metrics(coordinates, transformed)
    assert metrics["ca_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["ca_coverage"] == 1.0
    assert metrics["matched_ca_atoms"] == 5


def test_ca_metrics_length_mismatch() -> None:
    predicted = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    metrics = ca_metrics(predicted, reference)
    assert metrics["matched_ca_atoms"] == 3
    assert metrics["ca_coverage"] == pytest.approx(3 / 5)
    assert metrics["ca_match_mode"] == "truncated"


def test_ca_metrics_zero_atoms_raises() -> None:
    with pytest.raises(ValueError):
        ca_metrics(np.empty((0, 3)), np.empty((3, 3)))


def _build_structure_case(
    tmp_path: Path,
    *,
    predicted_lines: list[str],
    reference_lines: list[str],
    chain_map: dict[str, str] | None = None,
    reference_sha256: str | None = None,
    ca_coverage_threshold: float = 0.7,
) -> dict[str, object]:
    """Build a synthetic completed run and return the validate_run summary."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_pdb(run_dir / "AF-0000000000000001-model_v1.pdb", predicted_lines)
    reference_path = tmp_path / "ref.pdb"
    _write_pdb(reference_path, reference_lines)
    digest = reference_sha256 or hashlib.sha256(reference_path.read_bytes()).hexdigest()
    (run_dir / "AF-0000000000000001-meta_v1.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plddt": [0.9, 0.8],
                "pae": [[0.1, 0.2], [0.2, 0.1]],
                "max_pae": 0.2,
                "ptm": 0.5,
                "iptm": 0.4,
            }
        ),
        encoding="utf-8",
    )
    index = {
        "schema_version": 1,
        "run_id": "r",
        "entries": [
            {
                "target_id": "t1",
                "sequence_sha256": "a" * 64,
                "model_entity_id": "AF-0000000000000001",
                "tool_used": "OpenFold / AlphaFold-Multimer",
                "structure_path": "AF-0000000000000001-model_v1.pdb",
                "scores_path": "AF-0000000000000001-meta_v1.json",
            }
        ],
    }
    suite = {
        "schema_version": 1,
        "dataset_id": "x",
        "fingerprint": "f",
        "cases": [
            {
                "target_id": "t1",
                "sequence_sha256": "a" * 64,
                "thresholds": {"ca_coverage": ca_coverage_threshold},
                "require_no_nan": True,
                "expected_pair_mode": "single",
                "reference_structure": "ref.pdb",
                "reference_sha256": digest,
                "chain_map": chain_map or {},
                "metadata": {},
            }
        ],
    }
    index_path = tmp_path / "index.json"
    suite_path = tmp_path / "suite.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    return validate_run(run_dir, suite_path, index_path, tmp_path)


def _two_atoms() -> list[str]:
    return [
        _atom_line(1, 1, 11.104, 13.207, 10.000),
        _atom_line(2, 2, 12.000, 14.000, 11.000),
    ]


def _two_chain_atoms(first_chain: str, second_chain: str) -> list[str]:
    return [
        _atom_line(1, 1, 11.104, 13.207, 10.000, chain=first_chain),
        _atom_line(2, 2, 12.000, 14.000, 11.000, chain=first_chain),
        _atom_line(3, 1, 5.0, 5.0, 5.0, chain=second_chain),
        _atom_line(4, 2, 6.0, 6.0, 6.0, chain=second_chain),
    ]


def test_structure_valid_chain_map_identity(tmp_path: Path) -> None:
    result = _build_structure_case(
        tmp_path,
        predicted_lines=_two_atoms(),
        reference_lines=_two_atoms(),
        chain_map={"A": "A"},
    )
    case = result["cases"][0]
    assert case["structure_valid"] is True
    assert case["ca_coverage"] == 1.0
    assert case["ca_rmsd"] == 0.0
    assert case["passed"] is True


def test_structure_valid_chain_map_renamed(tmp_path: Path) -> None:
    reference_lines = [
        _atom_line(1, 1, 11.104, 13.207, 10.000, chain="B"),
        _atom_line(2, 2, 12.000, 14.000, 11.000, chain="B"),
    ]
    result = _build_structure_case(
        tmp_path,
        predicted_lines=_two_atoms(),
        reference_lines=reference_lines,
        chain_map={"A": "B"},
    )
    case = result["cases"][0]
    assert case["structure_valid"] is True
    assert case["ca_coverage"] == 1.0


def test_structure_valid_chain_map_two_chain_bijection(tmp_path: Path) -> None:
    result = _build_structure_case(
        tmp_path,
        predicted_lines=_two_chain_atoms("A", "B"),
        reference_lines=_two_chain_atoms("X", "Y"),
        chain_map={"A": "X", "B": "Y"},
    )
    case = result["cases"][0]
    assert case["structure_valid"] is True
    assert case["ca_coverage"] == 1.0
    assert case["ca_rmsd"] == 0.0
    assert case["passed"] is True


def test_structure_chain_map_omitting_a_predicted_chain_fails(tmp_path: Path) -> None:
    result = _build_structure_case(
        tmp_path,
        predicted_lines=_two_chain_atoms("A", "B"),
        reference_lines=_two_chain_atoms("A", "B"),
        chain_map={"A": "A"},
    )
    case = result["cases"][0]
    assert case["structure_valid"] is False
    assert case["ca_coverage"] is None
    assert case["ca_rmsd"] is None
    assert case["passed"] is False


def test_structure_chain_map_leaving_a_reference_chain_unmapped_fails(tmp_path: Path) -> None:
    result = _build_structure_case(
        tmp_path,
        predicted_lines=_two_atoms(),
        reference_lines=_two_chain_atoms("A", "B"),
        chain_map={"A": "A"},
    )
    case = result["cases"][0]
    assert case["structure_valid"] is False
    assert case["passed"] is False


def test_structure_chain_map_with_duplicate_reference_target_fails(tmp_path: Path) -> None:
    result = _build_structure_case(
        tmp_path,
        predicted_lines=_two_chain_atoms("A", "B"),
        reference_lines=_two_atoms(),
        chain_map={"A": "A", "B": "A"},
    )
    case = result["cases"][0]
    assert case["structure_valid"] is False
    assert case["passed"] is False


def test_structure_valid_reference_sha256_mismatch(tmp_path: Path) -> None:
    result = _build_structure_case(
        tmp_path,
        predicted_lines=_two_atoms(),
        reference_lines=_two_atoms(),
        chain_map={"A": "A"},
        reference_sha256="b" * 64,
    )
    case = result["cases"][0]
    assert case["structure_valid"] is False
    assert case["ca_coverage"] is None
    assert case["ca_rmsd"] is None
    assert case["passed"] is False


def test_structure_valid_coverage_below_threshold(tmp_path: Path) -> None:
    reference_lines = [
        _atom_line(1, 1, 11.104, 13.207, 10.000),
        _atom_line(2, 2, 12.000, 14.000, 11.000),
        _atom_line(3, 3, 13.000, 15.000, 12.000),
        _atom_line(4, 4, 14.000, 16.000, 13.000),
    ]
    result = _build_structure_case(
        tmp_path,
        predicted_lines=_two_atoms(),
        reference_lines=reference_lines,
        chain_map={"A": "A"},
    )
    case = result["cases"][0]
    assert case["structure_valid"] is False
    assert case["ca_coverage"] == pytest.approx(0.5)
    assert case["ca_rmsd"] == 0.0
    assert case["passed"] is False
