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

"""End-to-end composition + validity gate glue.

This module is the offline composition surface for the joint-phase
end-to-end gate. It composes the three phase seams — preprocessing MSA-set
identity, folding canonical pairs judged by Track C's ``validate_run``, and
the terminal postprocessing acceptance wrappers — into a bounded
``composition+validity`` record. It applies the shipped validation-suite
criteria VERBATIM (``prediction_count >= 1``, ``ca_coverage >= 0.7``, finite
output, no NaN) and never invents a structural-accuracy threshold
(``ca_rmsd`` or any pLDDT/confidence bound is deliberately absent).

The module is pure and offline: it imports only the runtime Track C benchmark
modules plus the standard library, renders (never submits) the frozen
postprocessing acceptance wrappers, and introduces no coordinator, intent,
lock, checkpoint, handoff, scheduler-evidence, or diagnostics machinery.
It must not import any Control Plane module (the runtime
import boundary).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

from bspp.orchestration.runtime.folding.benchmark.index import (
    CanonicalPairIndex,
    load_canonical_pair_index,
)
from bspp.orchestration.runtime.folding.benchmark.validation import validate_run

_MSA_SET_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")
_VALID_MATCH_MODES = {"by-tar", "aggregate"}

_FINITE_OUTPUT_FIELDS = ("mean_plddt", "min_plddt", "max_plddt", "plddt_above_70", "max_pae")


def _is_finite_number(value: object) -> bool:
    """Return ``True`` for a finite ``int``/``float`` (``bool`` excluded)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _verbatim_case(case: Mapping[str, object]) -> dict[str, object]:
    """Apply the four shipped criteria verbatim to one validation case.

    The returned projection is bounded and deliberately omits ``ca_rmsd`` and
    every confidence/pLDDT threshold: this gate is composition + validity, never
    structural accuracy.
    """
    raw_prediction_count = case.get("prediction_count")
    prediction_count: int | None = (
        raw_prediction_count
        if isinstance(raw_prediction_count, int) and not isinstance(raw_prediction_count, bool)
        else None
    )
    prediction_count_ok = prediction_count is not None and prediction_count >= 1

    raw_ca_coverage = case.get("ca_coverage")
    ca_coverage: float | None = raw_ca_coverage if isinstance(raw_ca_coverage, float) else None
    ca_coverage_ok = ca_coverage is not None and ca_coverage >= 0.7

    finite_output_ok = all(_is_finite_number(case.get(field)) for field in _FINITE_OUTPUT_FIELDS)
    no_nan_ok = case.get("output_has_nan") is False
    passed = case.get("passed") is True
    criteria_ok = prediction_count_ok and ca_coverage_ok and finite_output_ok and no_nan_ok

    return {
        "prediction_count": prediction_count,
        "prediction_count_ok": prediction_count_ok,
        "ca_coverage": ca_coverage,
        "ca_coverage_ok": ca_coverage_ok,
        "finite_output_ok": finite_output_ok,
        "no_nan_ok": no_nan_ok,
        "passed": passed,
        "criteria_ok": criteria_ok,
    }


def _require_validation_result_fields(
    validation_result: Mapping[str, object],
) -> tuple[str, str, int, int, list[Mapping[str, object]]]:
    """Extract the required bounded ``validate_run`` summary fields, fail closed."""
    dataset_id = validation_result.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise ValueError("validation_result must carry a string dataset_id")
    fingerprint = validation_result.get("fingerprint")
    if not isinstance(fingerprint, str):
        raise ValueError("validation_result must carry a string fingerprint")
    case_count = validation_result.get("case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int):
        raise ValueError("validation_result must carry an integer case_count")
    passed_count = validation_result.get("passed_count")
    if isinstance(passed_count, bool) or not isinstance(passed_count, int):
        raise ValueError("validation_result must carry an integer passed_count")
    raw_cases = validation_result.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("validation_result must carry a non-empty cases list")
    cases: list[Mapping[str, object]] = []
    for index, item in enumerate(raw_cases):
        if not isinstance(item, Mapping):
            raise ValueError(f"validation_result cases[{index}] must be a mapping")
        cases.append(item)
    # Internal consistency, fail closed: the declared counts must be derivable
    # from the case records, so a truncated or edited summary cannot pass.
    if len(cases) != case_count:
        raise ValueError(f"validation_result case_count {case_count} does not match the {len(cases)} case record(s)")
    recomputed_passed = sum(1 for case in cases if case.get("passed") is True)
    if recomputed_passed != passed_count:
        raise ValueError(
            f"validation_result passed_count {passed_count} does not match "
            f"the {recomputed_passed} passed case record(s)"
        )
    return dataset_id, fingerprint, case_count, passed_count, cases


def build_e2e_composition_evidence(
    *,
    canonical_pair_index: CanonicalPairIndex,
    validation_result: Mapping[str, object],
    preprocessing_msa_set_id: str | None = None,
    postprocessing_terminal_judge: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compose the three phase seams into bounded composition+validity evidence.

    ``canonical_pair_index`` (required) supplies the folding seam; the
    preprocessing seam is the optional content-addressed MSA-set id; the
    postprocessing seam is the optional terminal judge (the output of
    :func:`render_acceptance_wrapper_invocation` is the intended input).
    ``validity_ok`` applies the shipped criteria verbatim to every case, and
    ``accepted`` additionally requires both outer seams to be present.
    """
    dataset_id, fingerprint, case_count, passed_count, cases = _require_validation_result_fields(validation_result)

    if preprocessing_msa_set_id is not None:
        if _MSA_SET_ID_RE.fullmatch(preprocessing_msa_set_id) is None:
            raise ValueError("preprocessing_msa_set_id must be sha256:<64 lowercase hex>")
        preprocessing_seam: dict[str, object] | None = {"msa_set_id": preprocessing_msa_set_id}
    else:
        preprocessing_seam = None

    folding_seam: dict[str, object] = {
        "run_id": canonical_pair_index.run_id,
        "entry_count": len(canonical_pair_index.entries),
        "target_ids": [entry.target_id for entry in canonical_pair_index.entries],
    }

    projected_cases = [_verbatim_case(case) for case in cases]
    validity_ok = passed_count == case_count and all(
        case["criteria_ok"] is True and case["passed"] is True for case in projected_cases
    )
    composition_ok = preprocessing_seam is not None and postprocessing_terminal_judge is not None

    return {
        "schema_version": 1,
        "kind": "e2e-composition-validity",
        "gate": "composition+validity",
        "seams": {
            "preprocessing": preprocessing_seam,
            "folding": folding_seam,
            "postprocessing": postprocessing_terminal_judge,
        },
        "criteria": {
            "prediction_count_min": 1,
            "ca_coverage_min": 0.7,
            "finite_output_required": True,
            "no_nan_required": True,
            "note": "Track C shipped criteria applied verbatim; no RMSD/confidence threshold",
        },
        "validation": {
            "dataset_id": dataset_id,
            "fingerprint": fingerprint,
            "case_count": case_count,
            "passed_count": passed_count,
            "cases": projected_cases,
        },
        "validity_ok": validity_ok,
        "composition_ok": composition_ok,
        "accepted": validity_ok and composition_ok,
    }


def validate_completed_folding_run(
    *,
    run_root: Path,
    index_path: Path,
    suite_path: Path,
    corpus_dir: Path,
    output_dir: Path | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, object]:
    """Load a completed run's canonical-pair index and invoke Track C ``validate_run``.

    The explicit index load is the documented fail-closed gate:
    an absent/unreadable/malformed/unknown-schema/duplicate-target index raises
    ``ValueError`` before any per-case work, and the index is never reconstructed
    by discovery. ``validate_run`` re-loads the index internally; that second
    load is intentional and cheap.
    """
    if not isinstance(run_root, Path):
        raise ValueError("run_root must be a pathlib.Path")
    if not run_root.is_dir():
        raise ValueError(f"run_root must be an existing directory: {run_root}")
    if not isinstance(index_path, Path):
        raise ValueError("index_path must be a pathlib.Path")
    # The explicit load is the documented fail-closed gate;
    # validate_run re-loads the same index internally.
    _ = load_canonical_pair_index(index_path)
    return validate_run(
        run_root,
        suite_path,
        index_path,
        corpus_dir,
        output_dir=output_dir,
        expected_fingerprint=expected_fingerprint,
    )


def render_acceptance_wrapper_invocation(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    evidence_root: Path,
    verify_script: Path,
    relative_dir: str = "local_tars",
    baseline_run_name: str | None = None,
    candidate_run_name: str | None = None,
    workers: int = 1,
    payload_sample_count: int | None = None,
    match_mode: str = "by-tar",
    expected_tar_count: int | None = None,
    expected_local_tars_rows: int | None = None,
    expected_failed_rows: int | None = None,
    expected_analysis_rows: int | None = None,
    expected_selected_ids: int | None = None,
    parity_script: Path = Path("containers/scripts/slurm-tar-payload-parity.sh"),
    semantic_script: Path = Path("containers/scripts/slurm-semantic-acceptance.sh"),
    submitter_script: Path = Path("containers/scripts/submit-acceptance-checks.sh"),
) -> dict[str, object]:
    """Render (never submit) the frozen postprocessing acceptance wrapper invocations.

    The rendered argv maps to the ``validate tar-payload-parity`` and
    ``validate semantic-acceptance`` flags the wrappers forward via ``"$@"``,
    plus the ``submit-acceptance-checks.sh`` wiring. ``verify_script`` is the
    operator-supplied bounded evidence-verification wrapper required by the
    submitter; this glue never invents a verifier.
    """
    for name, path in (
        ("baseline_dir", baseline_dir),
        ("candidate_dir", candidate_dir),
        ("evidence_root", evidence_root),
    ):
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if match_mode not in _VALID_MATCH_MODES:
        raise ValueError(f"match_mode must be one of {sorted(_VALID_MATCH_MODES)!r}")
    if payload_sample_count is not None and payload_sample_count < 0:
        raise ValueError("payload_sample_count must be >= 0")

    parity_report_dir = evidence_root / "parity" / "tar_payload"
    parity_argv: list[str] = [
        str(parity_script),
        "--baseline-dir",
        str(baseline_dir),
        "--candidate-dir",
        str(candidate_dir),
        "--relative-dir",
        relative_dir,
        "--match-mode",
        match_mode,
        "--workers",
        str(workers),
        "--write-report",
        str(parity_report_dir),
        "--strict",
    ]
    if baseline_run_name is not None:
        parity_argv.extend(("--baseline-run-name", baseline_run_name))
    if candidate_run_name is not None:
        parity_argv.extend(("--candidate-run-name", candidate_run_name))
    if payload_sample_count is not None:
        parity_argv.extend(("--payload-sample-count", str(payload_sample_count)))

    semantic_report_dir = evidence_root / "semantic"
    semantic_argv: list[str] = [
        str(semantic_script),
        "--baseline-dir",
        str(baseline_dir),
        "--candidate-dir",
        str(candidate_dir),
        "--write-report",
        str(semantic_report_dir),
        "--strict",
    ]
    if expected_tar_count is not None:
        semantic_argv.extend(("--expected-tar-count", str(expected_tar_count)))
    if expected_local_tars_rows is not None:
        semantic_argv.extend(("--expected-local-tars-rows", str(expected_local_tars_rows)))
    if expected_failed_rows is not None:
        semantic_argv.extend(("--expected-failed-rows", str(expected_failed_rows)))
    if expected_analysis_rows is not None:
        semantic_argv.extend(("--expected-analysis-rows", str(expected_analysis_rows)))
    if expected_selected_ids is not None:
        semantic_argv.extend(("--expected-selected-ids", str(expected_selected_ids)))

    submitter_argv: list[str] = [
        str(submitter_script),
        "--parity-script",
        str(parity_script),
        "--semantic-script",
        str(semantic_script),
        "--verify-script",
        str(verify_script),
    ]

    return {
        "schema_version": 1,
        "kind": "postprocessing-acceptance-wrapper-invocation",
        "offline": True,
        "parity": {
            "wrapper": str(parity_script),
            "argv": parity_argv,
            "report_dir": str(parity_report_dir),
        },
        "semantic": {
            "wrapper": str(semantic_script),
            "argv": semantic_argv,
            "report_dir": str(semantic_report_dir),
        },
        "submitter": {
            "wrapper": str(submitter_script),
            "argv": submitter_argv,
            "never_submits": True,
        },
    }


__all__ = [
    "build_e2e_composition_evidence",
    "render_acceptance_wrapper_invocation",
    "validate_completed_folding_run",
]
