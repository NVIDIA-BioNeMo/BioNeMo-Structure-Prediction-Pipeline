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

"""Capture raw postprocessing acceptance evidence and adjudicate it once.

The legacy acceptance commands remain the producers of scientific reports.
This module never changes their exit codes or report bytes: ``capture`` records
them after structural validation, while ``adjudicate`` is the sole policy
decision for a postprocessing Phase attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from bspp.orchestration.contract.postprocessing_acceptance_adjudication import (
    PostprocessingAcceptanceAdjudication,
)
from bspp.orchestration.contract.postprocessing_acceptance_capture import (
    PostprocessingAcceptanceCapture,
    PostprocessingArtifactBinding,
    postprocessing_acceptance_capture_from_mapping,
)
from bspp.orchestration.contract.postprocessing_acceptance_diagnostics import (
    evaluate_postprocessing_acceptance_reports,
)
from bspp.orchestration.contract.postprocessing_acceptance_policy import (
    AcceptanceStepName,
    PostprocessingAcceptancePolicySnapshot,
    PostprocessingCompletionExitContract,
    postprocessing_acceptance_policy_from_mapping,
)
from bspp.orchestration.contract.postprocessing_attestations import (
    postprocessing_runtime_input_attestation_set_from_mapping,
)
from bspp.orchestration.contract.postprocessing_runspec import (
    postprocessing_phase_runspec_from_mapping,
)

_CAPTURE_STEPS: tuple[AcceptanceStepName, ...] = (
    "acceptance-tar-payload-parity",
    "acceptance-semantic",
    "acceptance-verify-evidence",
)
_SUPPORTED_REPORT_SCHEMAS = {
    ("tar-payload-parity-report", "1"),
    ("semantic-acceptance-summary", "1"),
    ("acceptance-evidence-report", "1"),
}


def capture_acceptance(
    *,
    policy_path: Path,
    expected_policy_sha256: str,
    evidence_root: Path,
    phase_run_id: str,
    attempt_id: str,
    action_id: str,
    step_name: AcceptanceStepName,
    raw_exit_code: int,
    raw_stdout_path: Path,
    raw_stderr_path: Path,
    output_path: Path,
    completed_at: str | None = None,
) -> PostprocessingAcceptanceCapture:
    """Validate and create-once bind the reports emitted by one raw command."""
    policy, policy_sha256 = _load_policy(policy_path, expected_policy_sha256)
    contract = _completion_contract(policy, step_name)
    if raw_exit_code not in contract.allowed_raw_exit_codes:
        raise ValueError(
            f"raw exit {raw_exit_code} is not permitted for {step_name}; allowed={contract.allowed_raw_exit_codes!r}"
        )

    reports: list[PostprocessingArtifactBinding] = []
    report_payloads: dict[str, Mapping[str, Any]] = {}
    for relative_path in contract.report_paths:
        report_path = _authority_relative_file(evidence_root, relative_path)
        report_bytes = report_path.read_bytes()
        payload = _json_mapping(report_bytes, label=f"acceptance report {relative_path!r}")
        _validate_report_schema(payload, contract=contract, relative_path=relative_path)
        report_payloads[relative_path] = payload
        reports.append(
            PostprocessingArtifactBinding(
                path=relative_path,
                sha256=hashlib.sha256(report_bytes).hexdigest(),
                size_bytes=len(report_bytes),
            )
        )
    expected_report_ok = {item.raw_exit_code: item.report_ok for item in contract.raw_exit_report_outcomes}[
        raw_exit_code
    ]
    observed_report_ok = _json_pointer(
        report_payloads[contract.outcome_report_path],
        contract.outcome_json_pointer,
    )
    if not isinstance(observed_report_ok, bool) or observed_report_ok is not expected_report_ok:
        raise ValueError(
            f"raw exit/report outcome mismatch for {step_name}: exit={raw_exit_code}, report_ok={observed_report_ok!r}"
        )
    raw_stdout = _stream_binding(evidence_root, raw_stdout_path, label="stdout")
    raw_stderr = _stream_binding(evidence_root, raw_stderr_path, label="stderr")

    capture = PostprocessingAcceptanceCapture(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        action_id=action_id,
        step_name=step_name,
        policy_sha256=policy_sha256,
        raw_exit_code=raw_exit_code,
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        reports=tuple(sorted(reports, key=lambda item: item.path)),
        completed_at=completed_at or _utc_timestamp(),
    )
    _write_create_once(output_path, _canonical_json_bytes(capture.to_mapping()))
    return capture


def adjudicate_acceptance(
    *,
    policy_path: Path,
    expected_policy_sha256: str,
    evidence_root: Path,
    phase_run_id: str,
    attempt_id: str,
    output_path: Path,
    adjudicated_at: str | None = None,
    phase_runspec_path: Path | None = None,
    expected_phase_runspec_digest: str | None = None,
    expected_baseline_locator: str | None = None,
) -> PostprocessingAcceptanceAdjudication:
    """Reconcile three immutable captures and emit the sole Phase verdict."""
    policy, policy_sha256 = _load_policy(policy_path, expected_policy_sha256)
    captures = tuple(
        _load_capture(
            evidence_root / "phase-acceptance" / f"{step_name}-capture.json",
            policy=policy,
            policy_sha256=policy_sha256,
            evidence_root=evidence_root,
            phase_run_id=phase_run_id,
            attempt_id=attempt_id,
            expected_step=step_name,
        )
        for step_name in _CAPTURE_STEPS
    )

    report_payloads: dict[str, Mapping[str, Any]] = {}
    for capture in captures:
        contract = _completion_contract(policy, capture.step_name)
        _verify_stream_binding(evidence_root, capture.raw_stdout, label="stdout")
        _verify_stream_binding(evidence_root, capture.raw_stderr, label="stderr")
        for report in capture.reports:
            report_path = _authority_relative_file(evidence_root, report.path)
            report_bytes = report_path.read_bytes()
            if len(report_bytes) != report.size_bytes or hashlib.sha256(report_bytes).hexdigest() != report.sha256:
                raise ValueError(f"captured acceptance report changed: {report.path!r}")
            payload = _json_mapping(report_bytes, label=f"acceptance report {report.path!r}")
            _validate_report_schema(payload, contract=contract, relative_path=report.path)
            previous = report_payloads.setdefault(report.path, payload)
            if previous != payload:
                raise ValueError(f"conflicting captured report payload: {report.path!r}")

    baseline_locator = _baseline_locator(
        evidence_root=evidence_root,
        phase_runspec_path=phase_runspec_path,
        expected_phase_runspec_digest=expected_phase_runspec_digest,
        expected_baseline_locator=expected_baseline_locator,
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
    )
    structural_errors = _baseline_binding_errors(policy, report_payloads, baseline_locator=baseline_locator)
    evaluation = evaluate_postprocessing_acceptance_reports(policy, report_payloads)
    non_allowlisted_errors = len(evaluation.unallowlisted_occurrences) + structural_errors
    residual_tuple = evaluation.residual_cardinalities
    reconciliation_tuple = evaluation.reconciliation_results
    residuals_ok = all(item.accepted for item in residual_tuple)
    reconciled = bool(reconciliation_tuple) and all(item.matched for item in reconciliation_tuple)
    result: Literal["passed", "failed"] = (
        "passed" if non_allowlisted_errors == 0 and residuals_ok and reconciled else "failed"
    )
    adjudication = PostprocessingAcceptanceAdjudication(
        phase_run_id=phase_run_id,
        attempt_id=attempt_id,
        policy_id=policy.policy_id,
        policy_sha256=policy_sha256,
        capture_digests=tuple(capture.digest for capture in captures),
        non_allowlisted_errors=non_allowlisted_errors,
        residual_cardinalities=residual_tuple,
        reconciliation_results=reconciliation_tuple,
        adjudicated_at=adjudicated_at or _utc_timestamp(),
        result=result,
    )
    _write_create_once(output_path, _canonical_json_bytes(adjudication.to_mapping()))
    return adjudication


def _load_capture(
    path: Path,
    *,
    policy: PostprocessingAcceptancePolicySnapshot,
    policy_sha256: str,
    evidence_root: Path,
    phase_run_id: str,
    attempt_id: str,
    expected_step: AcceptanceStepName,
) -> PostprocessingAcceptanceCapture:
    payload = _json_mapping(path.read_bytes(), label=f"acceptance capture {path}")
    capture = postprocessing_acceptance_capture_from_mapping(payload)
    if (
        capture.phase_run_id != phase_run_id
        or capture.attempt_id != attempt_id
        or capture.step_name != expected_step
        or capture.policy_sha256 != policy_sha256
    ):
        raise ValueError(f"acceptance capture identity mismatch: {path}")
    contract = _completion_contract(policy, expected_step)
    if capture.raw_exit_code not in contract.allowed_raw_exit_codes:
        raise ValueError(f"captured raw exit is forbidden by policy: {expected_step}")
    _verify_stream_binding(evidence_root, capture.raw_stdout, label="stdout")
    _verify_stream_binding(evidence_root, capture.raw_stderr, label="stderr")
    expected_paths = tuple(sorted(contract.report_paths))
    if tuple(item.path for item in capture.reports) != expected_paths:
        raise ValueError(f"acceptance capture report set differs from policy: {expected_step}")
    for report in capture.reports:
        report_bytes = _authority_relative_file(evidence_root, report.path).read_bytes()
        if hashlib.sha256(report_bytes).hexdigest() != report.sha256 or len(report_bytes) != report.size_bytes:
            raise ValueError(f"captured acceptance report changed: {report.path!r}")
    outcome_bytes = _authority_relative_file(evidence_root, contract.outcome_report_path).read_bytes()
    outcome_payload = _json_mapping(outcome_bytes, label=f"acceptance outcome report {expected_step}")
    expected_report_ok = {item.raw_exit_code: item.report_ok for item in contract.raw_exit_report_outcomes}[
        capture.raw_exit_code
    ]
    if _json_pointer(outcome_payload, contract.outcome_json_pointer) is not expected_report_ok:
        raise ValueError(f"captured raw exit/report outcome differs from policy: {expected_step}")
    return capture


def _load_policy(
    path: Path,
    expected_sha256: str,
) -> tuple[PostprocessingAcceptancePolicySnapshot, str]:
    policy_bytes = path.read_bytes()
    observed_sha256 = hashlib.sha256(policy_bytes).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("staged acceptance policy digest differs from the Phase RunSpec")
    payload = _json_mapping(policy_bytes, label="postprocessing acceptance policy")
    return postprocessing_acceptance_policy_from_mapping(payload), observed_sha256


def _completion_contract(
    policy: PostprocessingAcceptancePolicySnapshot,
    step_name: AcceptanceStepName,
) -> PostprocessingCompletionExitContract:
    return next(item for item in policy.completion_exit_contracts if item.step_name == step_name)


def _validate_report_schema(
    payload: Mapping[str, Any],
    *,
    contract: PostprocessingCompletionExitContract,
    relative_path: str,
) -> None:
    identity = (contract.report_schema, contract.report_schema_version)
    if identity not in _SUPPORTED_REPORT_SCHEMAS:
        raise ValueError(f"unsupported acceptance report schema identity: {identity!r}")
    if contract.report_schema == "tar-payload-parity-report":
        _require_keys(
            payload,
            {
                "baseline_dir",
                "candidate_dir",
                "ok",
                "inventory_errors",
                "baseline_only_tars",
                "candidate_only_tars",
                "payload_mismatch_count",
                "error_count",
                "files",
            },
            relative_path,
        )
        _require_type(payload, "ok", bool, relative_path)
        _require_type(payload, "payload_mismatch_count", int, relative_path)
        _require_type(payload, "error_count", int, relative_path)
        for key in ("inventory_errors", "baseline_only_tars", "candidate_only_tars", "files"):
            _require_type(payload, key, list, relative_path)
    elif contract.report_schema == "semantic-acceptance-summary":
        _require_keys(payload, {"baseline_dir", "candidate_dir", "ok", "errors"}, relative_path)
        _require_type(payload, "ok", bool, relative_path)
        _require_type(payload, "errors", list, relative_path)
    else:
        _require_keys(
            payload,
            {"schema_version", "ok", "parity_report_path", "semantic_report_path", "issues"},
            relative_path,
        )
        if payload["schema_version"] != 1:
            raise ValueError(f"acceptance report {relative_path!r} has the wrong embedded schema version")
        _require_type(payload, "ok", bool, relative_path)
        _require_type(payload, "issues", list, relative_path)


def _baseline_binding_errors(
    policy: PostprocessingAcceptancePolicySnapshot,
    reports: Mapping[str, Mapping[str, Any]],
    *,
    baseline_locator: str,
) -> int:
    errors = 0
    for binding in policy.baseline_report_bindings:
        try:
            payload = reports[binding.report]
            observed = _json_pointer(payload, binding.baseline_locator_json_pointer)
        except (KeyError, IndexError, TypeError, ValueError):
            errors += 1
            continue
        if observed != baseline_locator:
            errors += 1
    return errors


def _baseline_locator(
    *,
    evidence_root: Path,
    phase_runspec_path: Path | None,
    expected_phase_runspec_digest: str | None,
    expected_baseline_locator: str | None,
    phase_run_id: str,
    attempt_id: str,
) -> str:
    if expected_baseline_locator is not None:
        if phase_runspec_path is not None or expected_phase_runspec_digest is not None:
            raise ValueError("baseline locator test authority conflicts with Phase RunSpec authority")
        return expected_baseline_locator
    if phase_runspec_path is None or expected_phase_runspec_digest is None:
        raise ValueError("acceptance adjudication requires the frozen postprocessing Phase RunSpec")
    document = phase_runspec_path.read_bytes()
    payload = _json_mapping(document, label="postprocessing Phase RunSpec")
    runspec = postprocessing_phase_runspec_from_mapping(payload)
    if (
        runspec.digest != expected_phase_runspec_digest
        or runspec.phase_run_id != phase_run_id
        or runspec.attempt_id != attempt_id
    ):
        raise ValueError("acceptance adjudication Phase RunSpec identity differs")
    physical = {item.name: item for item in runspec.payload.physical_inputs}
    baseline = physical.get("baseline-output")
    if baseline is None:
        raise ValueError("postprocessing Phase RunSpec omits the baseline-output locator")
    attestations = postprocessing_runtime_input_attestation_set_from_mapping(
        _json_mapping(
            (evidence_root / "phase-inputs/runtime-input-attestations.json").read_bytes(),
            label="runtime input attestations",
        )
    )
    declared = next(
        (item for item in attestations.attestations if item.logical_input_name == "baseline-output"),
        None,
    )
    if (
        attestations.phase_run_id != phase_run_id
        or attestations.attempt_id != attempt_id
        or attestations.phase_runspec_digest != runspec.digest
        or declared is None
        or declared.physical_input_name != "baseline-output"
        or declared.physical_locator != baseline.locator
        or not declared.accessible
    ):
        raise ValueError("runtime baseline accessibility attestation differs from the Phase RunSpec")
    return baseline.locator


def _json_pointer(payload: object, pointer: str) -> object:
    if pointer == "":
        return payload
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be absolute")
    current = payload
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit():
                raise ValueError("JSON pointer list token must be an index")
            current = current[int(token)]
        else:
            raise TypeError("JSON pointer traverses a scalar")
    return current


def _authority_relative_file(root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ValueError(f"acceptance report path is not authority-relative: {relative_path!r}")
    resolved_root = root.resolve(strict=True)
    candidate = root / relative_path
    resolved = candidate.resolve(strict=True)
    if resolved_root not in resolved.parents or not resolved.is_file() or candidate.is_symlink():
        raise ValueError(f"acceptance report path escapes authority or is not a regular file: {relative_path!r}")
    return resolved


def _stream_binding(root: Path, path: Path, *, label: str) -> PostprocessingArtifactBinding:
    resolved = path.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if resolved_root not in resolved.parents or path.is_symlink() or not resolved.is_file():
        raise ValueError(f"raw {label} path escapes evidence root or is not a regular file")
    relative = resolved.relative_to(resolved_root).as_posix()
    data = resolved.read_bytes()
    return PostprocessingArtifactBinding(
        path=relative,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _verify_stream_binding(root: Path, binding: PostprocessingArtifactBinding, *, label: str) -> None:
    data = _authority_relative_file(root, binding.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != binding.sha256 or len(data) != binding.size_bytes:
        raise ValueError(f"captured raw {label} stream changed")


def _json_mapping(data: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return cast("Mapping[str, Any]", payload)


def _require_keys(payload: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(payload))
    if missing:
        raise ValueError(f"acceptance report {label!r} is missing schema fields: {missing!r}")


def _require_type(payload: Mapping[str, Any], key: str, expected: type[object], label: str) -> None:
    value = payload[key]
    if expected is int and isinstance(value, bool):
        raise TypeError(f"acceptance report {label!r} field {key!r} has the wrong type")
    if not isinstance(value, expected):
        raise TypeError(f"acceptance report {label!r} field {key!r} has the wrong type")


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def _write_create_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "adjudicate"):
        command = commands.add_parser(name)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--expected-policy-sha256", required=True)
        command.add_argument("--evidence-root", type=Path, required=True)
        command.add_argument("--phase-run-id", required=True)
        command.add_argument("--attempt-id", required=True)
        command.add_argument("--output", type=Path, required=True)
    capture = commands.choices["capture"]
    capture.add_argument("--action-id", required=True)
    capture.add_argument("--step-name", choices=_CAPTURE_STEPS, required=True)
    capture.add_argument("--raw-exit-code", type=int, required=True)
    capture.add_argument("--raw-stdout", type=Path, required=True)
    capture.add_argument("--raw-stderr", type=Path, required=True)
    adjudicate = commands.choices["adjudicate"]
    adjudicate.add_argument("--phase-runspec", type=Path, required=True)
    adjudicate.add_argument("--expected-phase-runspec-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "policy_path": args.policy,
        "expected_policy_sha256": args.expected_policy_sha256,
        "evidence_root": args.evidence_root,
        "phase_run_id": args.phase_run_id,
        "attempt_id": args.attempt_id,
        "output_path": args.output,
    }
    if args.command == "capture":
        capture_acceptance(
            **common,
            action_id=args.action_id,
            step_name=cast("AcceptanceStepName", args.step_name),
            raw_exit_code=args.raw_exit_code,
            raw_stdout_path=args.raw_stdout,
            raw_stderr_path=args.raw_stderr,
        )
        return 0
    adjudication = adjudicate_acceptance(
        **common,
        phase_runspec_path=args.phase_runspec,
        expected_phase_runspec_digest=args.expected_phase_runspec_digest,
    )
    return 0 if adjudication.result == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
