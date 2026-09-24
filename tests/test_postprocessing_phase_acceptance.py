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

"""Runtime capture and sole-adjudicator tests for postprocessing phases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bspp.orchestration.contract.postprocessing_acceptance import (
    PostprocessingAcceptancePolicySnapshot,
    PostprocessingBaselineReportBinding,
    PostprocessingCompletionExitContract,
    PostprocessingCrossReportReconciliation,
    PostprocessingRawExitReportOutcome,
    PostprocessingResidualAllowance,
)
from bspp.orchestration.contract.postprocessing_acceptance_diagnostics import (
    evaluate_postprocessing_acceptance_reports,
    project_unallowlisted_occurrences,
    safe_canonical_json_projection,
)
from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_phase_ids import postprocessing_action_id
from bspp.orchestration.runtime.postprocessing.phase_acceptance import (
    adjudicate_acceptance,
    capture_acceptance,
)

PHASE_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"
ATTEMPT_ID = "attempt-0001"
NOW = "2026-09-03T12:00:00Z"


def test_three_raw_captures_are_reconciled_by_the_sole_adjudicator(tmp_path: Path) -> None:
    policy_path, digest, evidence = _fixture(tmp_path)
    captures = []
    for step_name in (
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    ):
        capture = capture_acceptance(
            policy_path=policy_path,
            expected_policy_sha256=digest,
            evidence_root=evidence,
            phase_run_id=PHASE_RUN_ID,
            attempt_id=ATTEMPT_ID,
            action_id=postprocessing_action_id(step_name),
            step_name=step_name,
            raw_exit_code=0 if step_name == "acceptance-tar-payload-parity" else 1,
            raw_stdout_path=evidence / "phase-acceptance/raw" / f"{step_name}.stdout",
            raw_stderr_path=evidence / "phase-acceptance/raw" / f"{step_name}.stderr",
            output_path=evidence / "phase-acceptance" / f"{step_name}-capture.json",
            completed_at=NOW,
        )
        assert capture.raw_exit_code == (0 if step_name == "acceptance-tar-payload-parity" else 1)
        assert capture.raw_stdout.path.endswith(f"{step_name}.stdout")
        captures.append(capture)

    result = adjudicate_acceptance(
        policy_path=policy_path,
        expected_policy_sha256=digest,
        evidence_root=evidence,
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        output_path=evidence / "phase-acceptance" / "adjudication.json",
        adjudicated_at=NOW,
        expected_baseline_locator="/baseline",
    )

    assert result.result == "passed"
    assert result.non_allowlisted_errors == 0
    assert result.residual_cardinalities[0].observed == 1
    assert result.residual_cardinalities[0].permitted_min == 1
    assert result.residual_cardinalities[0].permitted_max == 1
    assert result.capture_digests == tuple(item.digest for item in captures)


def test_capture_rejects_raw_exit_or_report_schema_and_never_overwrites(tmp_path: Path) -> None:
    policy_path, digest, evidence = _fixture(tmp_path)
    output = evidence / "phase-acceptance" / "acceptance-semantic-capture.json"
    kwargs = {
        "policy_path": policy_path,
        "expected_policy_sha256": digest,
        "evidence_root": evidence,
        "phase_run_id": PHASE_RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "action_id": postprocessing_action_id("acceptance-semantic"),
        "step_name": "acceptance-semantic",
        "raw_stdout_path": evidence / "phase-acceptance/raw/acceptance-semantic.stdout",
        "raw_stderr_path": evidence / "phase-acceptance/raw/acceptance-semantic.stderr",
        "output_path": output,
        "completed_at": NOW,
    }
    with pytest.raises(ValueError, match="raw exit"):
        capture_acceptance(**kwargs, raw_exit_code=2)

    report = evidence / "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
    original = report.read_bytes()
    report.write_text(json.dumps({"ok": False, "errors": []}))
    with pytest.raises(ValueError, match="missing schema fields"):
        capture_acceptance(**kwargs, raw_exit_code=1)
    report.write_bytes(original)

    with pytest.raises(ValueError, match="raw exit/report outcome mismatch"):
        capture_acceptance(**kwargs, raw_exit_code=0)

    capture_acceptance(**kwargs, raw_exit_code=1)
    with pytest.raises(FileExistsError):
        capture_acceptance(**kwargs, raw_exit_code=1)


def test_adjudication_detects_report_drift_and_unallowlisted_errors(tmp_path: Path) -> None:
    policy_path, digest, evidence = _fixture(tmp_path)
    for step_name in (
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    ):
        capture_acceptance(
            policy_path=policy_path,
            expected_policy_sha256=digest,
            evidence_root=evidence,
            phase_run_id=PHASE_RUN_ID,
            attempt_id=ATTEMPT_ID,
            action_id=postprocessing_action_id(step_name),
            step_name=step_name,
            raw_exit_code=0 if step_name == "acceptance-tar-payload-parity" else 1,
            raw_stdout_path=evidence / "phase-acceptance/raw" / f"{step_name}.stdout",
            raw_stderr_path=evidence / "phase-acceptance/raw" / f"{step_name}.stderr",
            output_path=evidence / "phase-acceptance" / f"{step_name}-capture.json",
            completed_at=NOW,
        )
    semantic = evidence / "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
    payload = json.loads(semantic.read_text())
    payload["errors"].append("new-error")
    semantic.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ValueError, match="captured acceptance report changed"):
        adjudicate_acceptance(
            policy_path=policy_path,
            expected_policy_sha256=digest,
            evidence_root=evidence,
            phase_run_id=PHASE_RUN_ID,
            attempt_id=ATTEMPT_ID,
            output_path=evidence / "phase-acceptance" / "adjudication.json",
            adjudicated_at=NOW,
            expected_baseline_locator="/baseline",
        )


def test_adjudication_fails_for_wrong_baseline_locator_even_with_coherent_captures(tmp_path: Path) -> None:
    policy_path, digest, evidence = _fixture(tmp_path, baseline_locator="/wrong-baseline")
    for step_name in (
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    ):
        capture_acceptance(
            policy_path=policy_path,
            expected_policy_sha256=digest,
            evidence_root=evidence,
            phase_run_id=PHASE_RUN_ID,
            attempt_id=ATTEMPT_ID,
            action_id=postprocessing_action_id(step_name),
            step_name=step_name,
            raw_exit_code=0 if step_name == "acceptance-tar-payload-parity" else 1,
            raw_stdout_path=evidence / "phase-acceptance/raw" / f"{step_name}.stdout",
            raw_stderr_path=evidence / "phase-acceptance/raw" / f"{step_name}.stderr",
            output_path=evidence / "phase-acceptance" / f"{step_name}-capture.json",
            completed_at=NOW,
        )
    result = adjudicate_acceptance(
        policy_path=policy_path,
        expected_policy_sha256=digest,
        evidence_root=evidence,
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        output_path=evidence / "phase-acceptance" / "adjudication.json",
        adjudicated_at=NOW,
        expected_baseline_locator="/wrong-baseline",
    )
    assert result.result == "failed"
    assert result.non_allowlisted_errors == 2


def test_complete_acceptance_evaluator_keeps_secret_distinctions_before_bounded_projection(tmp_path: Path) -> None:
    policy_path, _digest, evidence = _fixture(tmp_path)
    policy = postprocessing_acceptance_policy_from_mapping(json.loads(policy_path.read_text()))
    parity_path = "acceptance/tar_payload_parity/tar_payload_parity_report.json"
    semantic_path = "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
    verify_path = "acceptance/verify_evidence/acceptance_evidence_report.json"
    semantic = json.loads((evidence / semantic_path).read_text())
    semantic["errors"] = [
        "known-residual",
        "https://user:secret-one@example.invalid/path?token=one",
        "https://user:secret-two@example.invalid/path?token=two",
        *[f"error-{index}" for index in range(64)],
    ]
    (evidence / semantic_path).write_text(json.dumps(semantic, sort_keys=True))
    reports = {path: json.loads((evidence / path).read_text()) for path in (parity_path, semantic_path, verify_path)}

    evaluation = evaluate_postprocessing_acceptance_reports(policy, reports)
    projection = project_unallowlisted_occurrences(evaluation.unallowlisted_occurrences)

    assert len(evaluation.unallowlisted_occurrences) == 66
    assert len(projection["rows"]) == 64
    assert projection["omitted_count"] == 2
    assert any("<redacted>" in row["value"] for row in projection["rows"])
    assert "secret-one" not in json.dumps(projection)
    assert "secret-two" not in json.dumps(projection)


def test_safe_canonical_projection_preserves_ordinary_json_and_uses_opaque_markers() -> None:
    assert safe_canonical_json_projection('"filename.tar"', max_display_value_chars=64) == '"filename.tar"'
    assert safe_canonical_json_projection("42", max_display_value_chars=64) == "42"
    assert safe_canonical_json_projection('{"count":1,"path":"a/b"}', max_display_value_chars=64) == (
        '{"count":1,"path":"a/b"}'
    )
    rendered = safe_canonical_json_projection(
        '{"https://user:key@example.invalid/?token=secret":"https://user:pass@example.invalid/?query=value"}',
        max_display_value_chars=256,
    )
    assert "user:key" not in rendered and "token=secret" not in rendered and "pass" not in rendered
    assert "<redacted>" in rendered
    assert safe_canonical_json_projection("not-json", max_display_value_chars=64).startswith("<opaque-canonical-json ")
    assert safe_canonical_json_projection('"' + "x" * 80 + '"', max_display_value_chars=8).startswith(
        "<opaque-canonical-json "
    )


def _fixture(tmp_path: Path, *, baseline_locator: str = "/baseline") -> tuple[Path, str, Path]:
    evidence = tmp_path / "evidence"
    parity_path = "acceptance/tar_payload_parity/tar_payload_parity_report.json"
    semantic_path = "acceptance/semantic_acceptance/semantic_acceptance_summary.json"
    verify_path = "acceptance/verify_evidence/acceptance_evidence_report.json"
    _write_json(
        evidence / parity_path,
        {
            "baseline_dir": "/baseline",
            "candidate_dir": "/candidate",
            "ok": True,
            "inventory_errors": [],
            "baseline_only_tars": [],
            "candidate_only_tars": [],
            "payload_mismatch_count": 0,
            "error_count": 0,
            "files": [],
        },
    )
    _write_json(
        evidence / semantic_path,
        {
            "baseline_dir": "/baseline",
            "candidate_dir": "/candidate",
            "ok": False,
            "errors": ["known-residual"],
        },
    )
    _write_json(
        evidence / verify_path,
        {
            "schema_version": 1,
            "ok": False,
            "parity_report_path": parity_path,
            "semantic_report_path": semantic_path,
            "issues": [{"check": "semantic-acceptance", "message": "ok is not true", "report_path": semantic_path}],
        },
    )
    for step_name in (
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    ):
        raw_root = evidence / "phase-acceptance/raw"
        raw_root.mkdir(parents=True, exist_ok=True)
        (raw_root / f"{step_name}.stdout").write_text(f"{step_name} stdout\n")
        (raw_root / f"{step_name}.stderr").write_text(f"{step_name} stderr\n")
    policy = PostprocessingAcceptancePolicySnapshot(
        baseline_id="baseline-853",
        baseline_version="v1",
        policy_schema="bspp-postprocessing-acceptance",
        policy_version="1",
        residual_allowances=(
            PostprocessingResidualAllowance(
                report=semantic_path,
                json_pointer="/errors",
                match_kind="json-pointer-count",
                expected_value="known-residual",
                cardinality_kind="exact",
                required_count=1,
                permitted_min=1,
                permitted_max=1,
            ),
            PostprocessingResidualAllowance(
                report=verify_path,
                json_pointer="/issues",
                match_kind="json-pointer-count",
                expected_value={
                    "check": "semantic-acceptance",
                    "message": "ok is not true",
                    "report_path": semantic_path,
                },
                cardinality_kind="exact",
                required_count=1,
                permitted_min=1,
                permitted_max=1,
            ),
        ),
        completion_exit_contracts=tuple(
            PostprocessingCompletionExitContract(
                step_name=step,
                allowed_raw_exit_codes=(0, 1),
                report_paths=(report,),
                report_schema=schema,
                report_schema_version="1",
                outcome_report_path=report,
                outcome_json_pointer="/ok",
                raw_exit_report_outcomes=(
                    PostprocessingRawExitReportOutcome(raw_exit_code=0, report_ok=True),
                    PostprocessingRawExitReportOutcome(raw_exit_code=1, report_ok=False),
                ),
            )
            for step, report, schema in (
                ("acceptance-tar-payload-parity", parity_path, "tar-payload-parity-report"),
                ("acceptance-semantic", semantic_path, "semantic-acceptance-summary"),
                ("acceptance-verify-evidence", verify_path, "acceptance-evidence-report"),
            )
        ),
        baseline_report_bindings=(
            PostprocessingBaselineReportBinding(
                report=parity_path,
                baseline_locator_json_pointer="/baseline_dir",
            ),
            PostprocessingBaselineReportBinding(
                report=semantic_path,
                baseline_locator_json_pointer="/baseline_dir",
            ),
        ),
        cross_report_reconciliations=(
            PostprocessingCrossReportReconciliation(
                left_report=parity_path,
                left_json_pointer="/candidate_dir",
                right_report=semantic_path,
                right_json_pointer="/candidate_dir",
            ),
        ),
    )
    policy_path = tmp_path / "acceptance-policy.json"
    _write_json(policy_path, policy.to_mapping())
    digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    return policy_path, digest, evidence


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
