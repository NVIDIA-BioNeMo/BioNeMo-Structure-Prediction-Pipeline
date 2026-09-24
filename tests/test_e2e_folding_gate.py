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

"""Tests for the e2e composition + validity gate glue (e08s06)."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

from bspp.orchestration.runtime.folding.benchmark.index import (
    build_canonical_pair_index,
)
from bspp.orchestration.runtime.folding.benchmark.validation import validate_run
from bspp.orchestration.runtime.folding.e2e_gate import (
    build_e2e_composition_evidence,
    render_acceptance_wrapper_invocation,
    validate_completed_folding_run,
)

_BENCHMARK_DIR = Path("folding") / "benchmark"
_INDEX_NAME = "validation-index.json"
_SUITE_NAME = "benchmark-suite.json"
_SCORES_NAME = "AF-0000000000000001-meta_v1.json"
_PREDICTED_STRUCTURE = "AF-0000000000000001-model_v1.pdb"
_REFERENCE_STRUCTURE = "refs/0001.pdb"

_PARITY_SCRIPT = "containers/scripts/slurm-tar-payload-parity.sh"
_SEMANTIC_SCRIPT = "containers/scripts/slurm-semantic-acceptance.sh"
_SUBMITTER_SCRIPT = "containers/scripts/submit-acceptance-checks.sh"


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


def _build_index() -> object:
    return build_canonical_pair_index(
        run_id="benchmark-fixture-run",
        entries=[
            (
                "pdb-temporal-2022-2025-v1/0001",
                "a" * 64,
                "AF-0000000000000001",
                "OpenFold / AlphaFold-Multimer",
                "AF-0000000000000001-model_v1.pdb",
                "AF-0000000000000001-meta_v1.json",
            )
        ],
    )


def _validation_result(**case_overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "target_id": "pdb-temporal-2022-2025-v1/0001",
        "identity_valid": True,
        "quality_valid": True,
        "structure_valid": True,
        "output_has_nan": False,
        "prediction_count": 1,
        "mean_plddt": 85.0,
        "min_plddt": 75.0,
        "max_plddt": 90.0,
        "plddt_above_70": 1.0,
        "max_pae": 3.46,
        "predicted_ca_atoms": 3,
        "reference_ca_atoms": 3,
        "matched_ca_atoms": 3,
        "ca_match_mode": "full",
        "ca_coverage": 1.0,
        "ca_rmsd": 0.0,
        "passed": True,
    }
    case.update(case_overrides)
    return {
        "schema_version": 1,
        "dataset_id": "pdb-temporal-2022-2025-v1",
        "fingerprint": "benchmark-fixture-fingerprint",
        "case_count": 1,
        "passed_count": 1,
        "cases": [case],
    }


def _msa_set_id() -> str:
    return "sha256:" + "0" * 64


def _postprocessing_judge() -> dict[str, object]:
    return {"kind": "postprocessing-acceptance-wrapper-invocation", "offline": True}


def _collect_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_collect_keys(item))
    return keys


def test_build_e2e_composition_evidence_verbatim_criteria(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    validation_result = validate_run(run_dir, suite_path, index_path, run_dir)
    evidence = build_e2e_composition_evidence(
        canonical_pair_index=_build_index(),
        validation_result=validation_result,
        preprocessing_msa_set_id=_msa_set_id(),
        postprocessing_terminal_judge=_postprocessing_judge(),
    )
    assert evidence["gate"] == "composition+validity"
    assert evidence["accepted"] is True
    assert evidence["validity_ok"] is True
    assert evidence["composition_ok"] is True

    seams = evidence["seams"]
    assert seams["preprocessing"] == {"msa_set_id": _msa_set_id()}
    assert seams["folding"]["run_id"] == "benchmark-fixture-run"
    assert seams["folding"]["entry_count"] == 1
    assert seams["folding"]["target_ids"] == ["pdb-temporal-2022-2025-v1/0001"]
    assert seams["postprocessing"] == _postprocessing_judge()

    criteria = evidence["criteria"]
    assert criteria["prediction_count_min"] == 1
    assert criteria["ca_coverage_min"] == 0.7
    assert criteria["finite_output_required"] is True
    assert criteria["no_nan_required"] is True

    case = evidence["validation"]["cases"][0]
    assert case["prediction_count"] == 1
    assert case["prediction_count_ok"] is True
    assert case["ca_coverage"] == 1.0
    assert case["ca_coverage_ok"] is True
    assert case["finite_output_ok"] is True
    assert case["no_nan_ok"] is True
    assert case["passed"] is True
    assert case["criteria_ok"] is True

    keys = _collect_keys(evidence)
    assert not any("rmsd" in key.lower() for key in keys)
    assert not any("confidence" in key.lower() for key in keys)


def test_build_e2e_composition_evidence_low_coverage_fails() -> None:
    evidence = build_e2e_composition_evidence(
        canonical_pair_index=_build_index(),
        validation_result=_validation_result(ca_coverage=0.5),
        preprocessing_msa_set_id=_msa_set_id(),
        postprocessing_terminal_judge=_postprocessing_judge(),
    )
    assert evidence["validity_ok"] is False
    assert evidence["accepted"] is False
    case = evidence["validation"]["cases"][0]
    assert case["ca_coverage_ok"] is False
    assert case["criteria_ok"] is False


def test_build_e2e_composition_evidence_nan_fails() -> None:
    evidence = build_e2e_composition_evidence(
        canonical_pair_index=_build_index(),
        validation_result=_validation_result(output_has_nan=True),
        preprocessing_msa_set_id=_msa_set_id(),
        postprocessing_terminal_judge=_postprocessing_judge(),
    )
    assert evidence["validity_ok"] is False
    assert evidence["accepted"] is False
    case = evidence["validation"]["cases"][0]
    assert case["no_nan_ok"] is False
    assert case["criteria_ok"] is False


def test_build_e2e_composition_evidence_zero_predictions_fails() -> None:
    evidence = build_e2e_composition_evidence(
        canonical_pair_index=_build_index(),
        validation_result=_validation_result(prediction_count=0),
        preprocessing_msa_set_id=_msa_set_id(),
        postprocessing_terminal_judge=_postprocessing_judge(),
    )
    assert evidence["validity_ok"] is False
    assert evidence["accepted"] is False
    case = evidence["validation"]["cases"][0]
    assert case["prediction_count"] == 0
    assert case["prediction_count_ok"] is False
    assert case["criteria_ok"] is False


def test_build_e2e_composition_evidence_requires_all_seams() -> None:
    validation_result = _validation_result()
    # Validity is ok with no outer seams, but composition is not.
    evidence = build_e2e_composition_evidence(
        canonical_pair_index=_build_index(),
        validation_result=validation_result,
    )
    assert evidence["validity_ok"] is True
    assert evidence["composition_ok"] is False
    assert evidence["accepted"] is False

    evidence = build_e2e_composition_evidence(
        canonical_pair_index=_build_index(),
        validation_result=validation_result,
        preprocessing_msa_set_id=_msa_set_id(),
    )
    assert evidence["composition_ok"] is False
    assert evidence["accepted"] is False

    evidence = build_e2e_composition_evidence(
        canonical_pair_index=_build_index(),
        validation_result=validation_result,
        postprocessing_terminal_judge=_postprocessing_judge(),
    )
    assert evidence["composition_ok"] is False
    assert evidence["accepted"] is False

    evidence = build_e2e_composition_evidence(
        canonical_pair_index=_build_index(),
        validation_result=validation_result,
        preprocessing_msa_set_id=_msa_set_id(),
        postprocessing_terminal_judge=_postprocessing_judge(),
    )
    assert evidence["composition_ok"] is True
    assert evidence["accepted"] is True


def test_e2e_gate_fails_closed_on_truncated_cases() -> None:
    # Declared counts claim two passing cases but only one record is present:
    # the summary is internally inconsistent and must not pass.
    result = _validation_result()
    result["case_count"] = 2
    result["passed_count"] = 2
    with pytest.raises(ValueError, match="case_count 2 does not match"):
        build_e2e_composition_evidence(
            canonical_pair_index=_build_index(),
            validation_result=result,
            preprocessing_msa_set_id=_msa_set_id(),
            postprocessing_terminal_judge=_postprocessing_judge(),
        )


def test_e2e_gate_fails_closed_on_edited_passed_count() -> None:
    # The declared passed_count disagrees with the case records (the single
    # case failed): recomputation must expose the edit and fail closed.
    result = _validation_result(passed=False)
    with pytest.raises(ValueError, match="passed_count 1 does not match"):
        build_e2e_composition_evidence(
            canonical_pair_index=_build_index(),
            validation_result=result,
            preprocessing_msa_set_id=_msa_set_id(),
            postprocessing_terminal_judge=_postprocessing_judge(),
        )


def test_validate_completed_folding_run_invokes_validate_run(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    result = validate_completed_folding_run(
        run_root=run_dir,
        index_path=index_path,
        suite_path=suite_path,
        corpus_dir=run_dir,
    )
    assert result["passed_count"] == 1
    assert result["cases"][0]["passed"] is True


def test_validate_completed_folding_run_fails_closed_on_absent_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_completed_folding_run(
            run_root=tmp_path,
            index_path=tmp_path / "missing-index.json",
            suite_path=tmp_path / "suite.json",
            corpus_dir=tmp_path,
        )


def test_validate_completed_folding_run_fails_closed_on_invalid_index(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_completed_folding_run(
            run_root=tmp_path,
            index_path=index_path,
            suite_path=tmp_path / "suite.json",
            corpus_dir=tmp_path,
        )


def test_validate_completed_folding_run_does_not_reconstruct_index(tmp_path: Path, fixtures_dir: Path) -> None:
    run_dir, _suite_path, index_path = _materialize_run(tmp_path, fixtures_dir)
    # The run directory contains the canonical pair files, but there is no
    # index anywhere: validation must fail closed rather than discover them.
    index_path.unlink()
    with pytest.raises(ValueError):
        validate_completed_folding_run(
            run_root=run_dir,
            index_path=index_path,
            suite_path=tmp_path / "suite.json",
            corpus_dir=run_dir,
        )


def test_validate_completed_folding_run_rejects_non_directory_run_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_completed_folding_run(
            run_root=tmp_path / "does-not-exist",
            index_path=tmp_path / "index.json",
            suite_path=tmp_path / "suite.json",
            corpus_dir=tmp_path,
        )


def test_render_acceptance_wrapper_invocation_renders_three_wrappers(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    evidence_root = tmp_path / "evidence"
    verify_script = tmp_path / "verify.sh"
    rendered = render_acceptance_wrapper_invocation(
        baseline_dir=baseline,
        candidate_dir=candidate,
        evidence_root=evidence_root,
        verify_script=verify_script,
        workers=4,
        payload_sample_count=10,
        expected_tar_count=5,
    )

    assert rendered["offline"] is True
    assert rendered["parity"]["wrapper"] == _PARITY_SCRIPT
    assert rendered["semantic"]["wrapper"] == _SEMANTIC_SCRIPT
    assert rendered["submitter"]["wrapper"] == _SUBMITTER_SCRIPT
    assert rendered["submitter"]["never_submits"] is True

    parity_argv = rendered["parity"]["argv"]
    assert "--match-mode" in parity_argv
    assert "by-tar" in parity_argv
    assert "--strict" in parity_argv
    assert "--write-report" in parity_argv
    assert "--payload-sample-count" in parity_argv
    assert "10" in parity_argv

    semantic_argv = rendered["semantic"]["argv"]
    assert "--strict" in semantic_argv
    assert "--expected-tar-count" in semantic_argv

    submitter_argv = rendered["submitter"]["argv"]
    assert "--parity-script" in submitter_argv
    assert "--semantic-script" in submitter_argv
    assert "--verify-script" in submitter_argv

    # Pure function: it never submits and never touches the filesystem.
    assert not any("sbatch" in arg for arg in parity_argv + semantic_argv + submitter_argv)
    assert list(tmp_path.iterdir()) == []


def test_render_acceptance_wrapper_invocation_fails_closed(tmp_path: Path) -> None:
    absolute = tmp_path / "abs"
    relative = Path("relative")
    with pytest.raises(ValueError):
        render_acceptance_wrapper_invocation(
            baseline_dir=relative,
            candidate_dir=absolute,
            evidence_root=absolute,
            verify_script=absolute,
        )
    with pytest.raises(ValueError):
        render_acceptance_wrapper_invocation(
            baseline_dir=absolute,
            candidate_dir=absolute,
            evidence_root=absolute,
            verify_script=absolute,
            match_mode="bogus",
        )
    with pytest.raises(ValueError):
        render_acceptance_wrapper_invocation(
            baseline_dir=absolute,
            candidate_dir=absolute,
            evidence_root=absolute,
            verify_script=absolute,
            workers=0,
        )
    with pytest.raises(ValueError):
        render_acceptance_wrapper_invocation(
            baseline_dir=absolute,
            candidate_dir=absolute,
            evidence_root=absolute,
            verify_script=absolute,
            payload_sample_count=-1,
        )
