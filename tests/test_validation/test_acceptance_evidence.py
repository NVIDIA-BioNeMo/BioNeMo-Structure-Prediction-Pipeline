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

"""Tests for lightweight acceptance evidence verification."""

from __future__ import annotations

import json
from pathlib import Path

from bspp.orchestration.runtime.validation.acceptance_evidence import verify_acceptance_evidence


def test_verify_acceptance_evidence_accepts_good_parity_and_semantic(tmp_path: Path) -> None:
    parity = _write_json(tmp_path / "tar_payload_parity_report.json", _good_parity())
    semantic = _write_json(tmp_path / "semantic_acceptance_summary.json", _good_semantic())

    report = verify_acceptance_evidence(parity_report_path=parity, semantic_report_path=semantic)

    assert report.ok is True
    assert report.issues == ()


def test_verify_acceptance_evidence_accepts_single_parity_report(tmp_path: Path) -> None:
    parity = _write_json(tmp_path / "tar_payload_parity_report.json", _good_parity())

    report = verify_acceptance_evidence(parity_report_path=parity)

    assert report.ok is True
    assert report.semantic_report_path is None
    assert report.to_redacted_dict()["parity_report_path"] == str(parity)


def test_verify_acceptance_evidence_rejects_no_report_paths() -> None:
    report = verify_acceptance_evidence()

    assert report.ok is False
    assert report.issues[0].check == "acceptance"
    assert report.issues[0].message == "at least one acceptance comparator report path is required"


def test_verify_acceptance_evidence_rejects_missing_expected_report(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    report = verify_acceptance_evidence(parity_report_path=missing)

    assert report.ok is False
    assert report.issues[0].message == "expected report is missing"
    assert report.issues[0].to_redacted_dict()["report_path"] == str(missing)


def test_verify_acceptance_evidence_rejects_inventory_only_parity(tmp_path: Path) -> None:
    payload = _good_parity()
    payload["payload_sample_count"] = 0
    parity = _write_json(tmp_path / "tar_payload_parity_report.json", payload)

    report = verify_acceptance_evidence(parity_report_path=parity)

    assert report.ok is False
    assert "inventory-only parity" in report.issues[0].message


def test_verify_acceptance_evidence_rejects_zero_compared_members(tmp_path: Path) -> None:
    payload = _good_parity()
    payload["compared_members"] = 0
    parity = _write_json(tmp_path / "tar_payload_parity_report.json", payload)

    report = verify_acceptance_evidence(parity_report_path=parity)

    assert report.ok is False
    assert "compared_members is not positive" in report.issues[0].message


def test_verify_acceptance_evidence_rejects_semantic_errors(tmp_path: Path) -> None:
    payload = _good_semantic()
    payload["errors"] = ["local_tars.csv semantic rows differ"]
    semantic = _write_json(tmp_path / "semantic_acceptance_summary.json", payload)

    report = verify_acceptance_evidence(semantic_report_path=semantic)

    assert report.ok is False
    assert "errors is not empty" in report.issues[0].message


def _good_parity() -> dict[str, object]:
    return {
        "ok": True,
        "payload_mismatch_count": 0,
        "error_count": 0,
        "compared_tar_count": 2,
        "compared_members": 12,
        "payload_sample_count": 6,
    }


def _good_semantic() -> dict[str, object]:
    return {"ok": True, "errors": []}


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path
