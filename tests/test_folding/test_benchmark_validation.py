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

"""Tests for validate_run's index-identity and raw-scores quality gate."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import bspp.orchestration.runtime.folding.benchmark.validation as validation_module
from bspp.orchestration.runtime.folding.benchmark.validation import validate_run

_BENCHMARK_DIR = Path("folding") / "benchmark"
_INDEX_NAME = "validation-index.json"
_SUITE_NAME = "benchmark-suite.json"
_SCORES_NAME = "AF-0000000000000001-meta_v1.json"
_PREDICTED_STRUCTURE = "AF-0000000000000001-model_v1.pdb"
_REFERENCE_STRUCTURE = "refs/0001.pdb"


def _materialize_run(tmp_path: Path, fixtures_dir: Path) -> tuple[Path, Path, Path]:
    """Copy the synthetic fixtures into tmp_path and return (run_dir, suite_path, index_path)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = fixtures_dir / _BENCHMARK_DIR
    structure = source / "structure"
    index_path = tmp_path / "index.json"
    suite_path = tmp_path / "suite.json"
    shutil.copy(source / _INDEX_NAME, index_path)
    shutil.copy(source / _SUITE_NAME, suite_path)
    shutil.copy(source / _SCORES_NAME, run_dir / _SCORES_NAME)
    shutil.copy(structure / "predicted.pdb", run_dir / _PREDICTED_STRUCTURE)
    reference_path = run_dir / _REFERENCE_STRUCTURE
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(structure / "reference.pdb", reference_path)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["cases"][0]["reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    return run_dir, suite_path, index_path


def test_passing_identity_and_quality(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is True
    assert case["quality_valid"] is True
    assert case["output_has_nan"] is False
    assert case["prediction_count"] >= 1
    assert case["passed"] is True
    assert result["passed_count"] == 1


@pytest.mark.parametrize("absolute_paths", [False, True])
def test_executor_index_paths_validate_without_rewriting_index(
    tmp_path: Path, fixtures_dir: Path, absolute_paths: bool
) -> None:
    """Consume the executor's golden index using only existing test fixture bytes."""
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    golden = fixtures_dir / "folding" / "executable" / "expected-canonical-pair-index.json"
    index_text = golden.read_text(encoding="utf-8").replace("/ATTEMPT_ROOT", str(run_dir))
    index = json.loads(index_text)
    entry = index["entries"][0]
    for field, source_name in (("structure_path", _PREDICTED_STRUCTURE), ("scores_path", _SCORES_NAME)):
        destination = Path(entry[field])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / source_name, destination)
        if not absolute_paths:
            entry[field] = str(destination.relative_to(run_dir))
    if not absolute_paths:
        index_text = json.dumps(index)
    index_path.write_text(index_text, encoding="utf-8")
    index_bytes = index_path.read_bytes()
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["cases"][0].update(target_id=entry["target_id"], sequence_sha256=entry["sequence_sha256"])
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    result = validate_run(run_dir, suite_path, index_path, run_dir)

    assert result["passed_count"] == 1
    case = result["cases"][0]
    assert case["identity_valid"] is True
    assert case["quality_valid"] is True
    assert case["structure_valid"] is True
    assert case["ca_coverage"] == 1.0
    assert case["ca_rmsd"] == 0.0
    assert index_path.read_bytes() == index_bytes
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == index_digest


@pytest.mark.parametrize("field", ["scores_path", "structure_path"])
@pytest.mark.parametrize("absolute_path", [False, True])
def test_prediction_symlink_escape_fails_closed(
    tmp_path: Path, fixtures_dir: Path, field: str, absolute_path: bool
) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifact = run_dir / index["entries"][0][field]
    outside = tmp_path / artifact.name
    shutil.copyfile(artifact, outside)
    artifact.unlink()
    artifact.symlink_to(outside)
    if absolute_path:
        index["entries"][0][field] = str(artifact)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    case = validate_run(run_dir, suite_path, index_path, run_dir)["cases"][0]

    assert case["identity_valid" if field == "scores_path" else "structure_valid"] is False
    assert case["passed"] is False


@pytest.mark.parametrize("field", ["scores_path", "structure_path"])
def test_absolute_prediction_parent_escape_fails_closed(tmp_path: Path, fixtures_dir: Path, field: str) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    name = index["entries"][0][field]
    shutil.copyfile(run_dir / name, tmp_path / name)
    index["entries"][0][field] = str(run_dir / ".." / name)
    index_path.write_text(json.dumps(index), encoding="utf-8")

    case = validate_run(run_dir, suite_path, index_path, run_dir)["cases"][0]

    assert case["identity_valid" if field == "scores_path" else "structure_valid"] is False
    assert case["passed"] is False


def test_contained_absolute_reference_remains_rejected(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["cases"][0]["reference_structure"] = str(run_dir / _REFERENCE_STRUCTURE)
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    case = validate_run(run_dir, suite_path, index_path, run_dir)["cases"][0]

    assert case["identity_valid"] is True
    assert case["structure_valid"] is False
    assert case["passed"] is False


def test_sequence_sha256_mismatch(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["sequence_sha256"] = "b" * 64
    index_path.write_text(json.dumps(index), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is False
    assert case["passed"] is False


def test_missing_target_id(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["target_id"] = "pdb-temporal-2022-2025-v1/9999"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is False
    assert case["passed"] is False


def test_resolve_entry_matches_pdb_assembly_target_id(tmp_path: Path, fixtures_dir: Path) -> None:
    """AC #20: _resolve_entry matches an index entry with target_id pdb_5snm_assembly_1
    and the correct sequence_sha256."""
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    pdb_target_id = "pdb_5snm_assembly_1"
    seq_sha = "a" * 64
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["target_id"] = pdb_target_id
    index["entries"][0]["sequence_sha256"] = seq_sha
    index_path.write_text(json.dumps(index), encoding="utf-8")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["cases"][0]["target_id"] = pdb_target_id
    suite["cases"][0]["sequence_sha256"] = seq_sha
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is True
    assert case["target_id"] == pdb_target_id


def test_malformed_scores_json(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    (run_dir / _SCORES_NAME).write_text("{not json", encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is False
    assert case["passed"] is False


def test_require_no_nan_on_nan_plddt(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    scores_path = run_dir / _SCORES_NAME
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    scores["plddt"] = [0.9, float("nan"), 0.8]
    scores_path.write_text(json.dumps(scores, allow_nan=True), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["output_has_nan"] is True
    assert case["quality_valid"] is False
    assert case["passed"] is False


def test_postprocess_check_dropped(tmp_path: Path, fixtures_dir: Path) -> None:
    source = inspect.getsource(validation_module)
    assert "_PostprocessSnapshot" not in source
    assert "postprocess_" not in source
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    assert result["cases"][0]["passed"] is True


def test_suite_fingerprint_binding_rejects_mismatch(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_run(run_dir, suite_path, index_path, run_dir, expected_fingerprint="0" * 64)


def test_suite_fingerprint_binding_accepts_match(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    result = validate_run(
        run_dir,
        suite_path,
        index_path,
        run_dir,
        expected_fingerprint="benchmark-fixture-fingerprint",
    )
    assert result["cases"][0]["passed"] is True


# The escaping copies keep the canonical AF-* file names and valid contents so
# that, without containment, the contract pair load and C-alpha alignment would
# succeed: only the containment guard may cause these failures.
def test_absolute_scores_path_fails_closed(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    outside = tmp_path / _SCORES_NAME
    shutil.copy(run_dir / _SCORES_NAME, outside)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["scores_path"] = str(outside)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is False
    assert case["passed"] is False


def test_parent_traversal_scores_path_fails_closed(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    outside = tmp_path / _SCORES_NAME
    shutil.copy(run_dir / _SCORES_NAME, outside)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["scores_path"] = f"../{_SCORES_NAME}"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is False
    assert case["passed"] is False


def test_absolute_structure_path_fails_closed(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    outside = tmp_path / _PREDICTED_STRUCTURE
    shutil.copy(run_dir / _PREDICTED_STRUCTURE, outside)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["structure_path"] = str(outside)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is True
    assert case["structure_valid"] is False
    assert case["ca_coverage"] is None
    assert case["passed"] is False


def test_parent_traversal_structure_path_fails_closed(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    outside = tmp_path / _PREDICTED_STRUCTURE
    shutil.copy(run_dir / _PREDICTED_STRUCTURE, outside)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][0]["structure_path"] = f"../{_PREDICTED_STRUCTURE}"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is True
    assert case["structure_valid"] is False
    assert case["ca_coverage"] is None
    assert case["passed"] is False


def test_absolute_reference_structure_fails_closed(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    outside = tmp_path / "outside.pdb"
    shutil.copy(run_dir / _REFERENCE_STRUCTURE, outside)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["cases"][0]["reference_structure"] = str(outside)
    # The bytes and digest still match: only containment may cause the failure.
    suite["cases"][0]["reference_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["structure_valid"] is False
    assert case["passed"] is False


def test_parent_traversal_reference_structure_fails_closed(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    outside = tmp_path / "outside.pdb"
    shutil.copy(run_dir / _REFERENCE_STRUCTURE, outside)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["cases"][0]["reference_structure"] = "../outside.pdb"
    suite["cases"][0]["reference_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    # corpus_dir=run_dir: run_dir/../outside.pdb resolves to the copied file,
    # so without containment the digest would match and the case would pass.
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["structure_valid"] is False
    assert case["passed"] is False


def test_end_to_end_structure_and_evidence(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    result = validate_run(run_dir, suite_path, index_path, run_dir)
    case = result["cases"][0]
    assert case["identity_valid"] is True
    assert case["quality_valid"] is True
    assert case["structure_valid"] is True
    assert case["ca_coverage"] == 1.0
    assert case["ca_rmsd"] == 0.0
    assert case["passed"] is True
    assert result["passed_count"] == 1
    assert result["dataset_id"] == "pdb-temporal-2022-2025-v1"
    assert result["fingerprint"] == "benchmark-fixture-fingerprint"

    parquet_path = run_dir / "validation" / "validation.parquet"
    summary_path = run_dir / "validation" / "summary.json"
    assert parquet_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["cases"][0]["structure_valid"] is True
    table = pq.read_table(parquet_path)
    assert table.column_names == [
        "target_id",
        "identity_valid",
        "quality_valid",
        "structure_valid",
        "ca_coverage",
        "ca_rmsd",
        "output_has_nan",
        "passed",
    ]
    assert table.num_rows == 1


@pytest.mark.parametrize("corruption", [None, "missing-pae", "nan-pae"])
def test_native_monomer_provenance_requires_real_finite_full_scores(
    tmp_path: Path, fixtures_dir: Path, corruption: str | None
) -> None:
    """New provenance and undefined ipTM do not relax the existing PAE gate."""
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    index = json.loads(index_path.read_bytes())
    index["entries"][0]["tool_used"] = "OpenFold2 (BioNeMo IR) / OpenFold-pTM"
    index_path.write_text(json.dumps(index))
    original_index = index_path.read_bytes()
    score_path = run_dir / _SCORES_NAME
    scores = json.loads(score_path.read_bytes())
    scores["iptm"] = None
    if corruption == "missing-pae":
        del scores["pae"]
    elif corruption == "nan-pae":
        scores["pae"][0][0] = float("nan")
    score_path.write_text(json.dumps(scores))

    result = validate_run(run_dir, suite_path, index_path, run_dir)

    assert result["cases"][0]["passed"] is (corruption is None)
    assert index_path.read_bytes() == original_index
