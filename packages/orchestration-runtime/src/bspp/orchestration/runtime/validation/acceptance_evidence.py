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

"""Lightweight JSON acceptance evidence verification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bspp.orchestration.contract.postprocessing_acceptance_diagnostics import (
    project_acceptance_evidence_issues,
)
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary


@dataclass(frozen=True)
class AcceptanceEvidenceIssue:
    """One acceptance evidence verification issue."""

    check: str
    report_path: Path | None
    message: str

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready issue data."""
        return {
            "check": self.check,
            "report_path": str(self.report_path) if self.report_path is not None else None,
            "message": self.message,
        }


@dataclass(frozen=True)
class AcceptanceEvidenceReport:
    """JSON-only acceptance evidence verification report."""

    schema_version: int
    parity_report_path: Path | None
    semantic_report_path: Path | None
    issues: tuple[AcceptanceEvidenceIssue, ...]

    @property
    def ok(self) -> bool:
        """Return true when all requested evidence checks passed."""
        return not self.issues

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready report data."""
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "parity_report_path": str(self.parity_report_path) if self.parity_report_path is not None else None,
            "semantic_report_path": str(self.semantic_report_path) if self.semantic_report_path is not None else None,
            "issues": [issue.to_redacted_dict() for issue in self.issues],
        }


def verify_acceptance_evidence(
    *,
    parity_report_path: Path | None = None,
    semantic_report_path: Path | None = None,
) -> AcceptanceEvidenceReport:
    """Verify small JSON reports from the enabled acceptance comparator steps."""
    issues: list[AcceptanceEvidenceIssue] = []
    if parity_report_path is None and semantic_report_path is None:
        issues.append(
            AcceptanceEvidenceIssue(
                check="acceptance",
                report_path=None,
                message="at least one acceptance comparator report path is required",
            )
        )

    if parity_report_path is not None:
        issues.extend(_verify_parity(parity_report_path))
    if semantic_report_path is not None:
        issues.extend(_verify_semantic(semantic_report_path))

    return AcceptanceEvidenceReport(
        schema_version=1,
        parity_report_path=parity_report_path,
        semantic_report_path=semantic_report_path,
        issues=tuple(issues),
    )


def render_acceptance_evidence_report(report: AcceptanceEvidenceReport) -> str:
    """Render deterministic JSON for acceptance evidence verification."""
    return report_to_json(report)


def write_acceptance_evidence_report(
    report: AcceptanceEvidenceReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON and text summaries for acceptance evidence verification."""
    json_path = write_json_report(report, output_dir / "acceptance_evidence_report.json")
    text_path = write_text_summary(report, output_dir / "acceptance_evidence_report.txt")
    return json_path, text_path


def _verify_parity(path: Path) -> tuple[AcceptanceEvidenceIssue, ...]:
    payload, issue = _read_json(path, "tar-payload-parity")
    if issue is not None:
        return (issue,)
    assert payload is not None

    return _issues_from_projection(
        project_acceptance_evidence_issues(parity_report=payload, parity_report_path=str(path))
    )


def _verify_semantic(path: Path) -> tuple[AcceptanceEvidenceIssue, ...]:
    payload, issue = _read_json(path, "semantic-acceptance")
    if issue is not None:
        return (issue,)
    assert payload is not None

    return _issues_from_projection(
        project_acceptance_evidence_issues(semantic_report=payload, semantic_report_path=str(path))
    )


def _read_json(path: Path, check: str) -> tuple[Mapping[str, Any] | None, AcceptanceEvidenceIssue | None]:
    if not path.exists():
        return None, _issue(check, path, "expected report is missing")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, _issue(check, path, f"report is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        return None, _issue(check, path, "report JSON must be an object")
    return payload, None


def _issues_from_projection(rows: tuple[dict[str, object], ...]) -> tuple[AcceptanceEvidenceIssue, ...]:
    return tuple(
        AcceptanceEvidenceIssue(
            check=str(row["check"]),
            report_path=Path(str(row["report_path"])) if row["report_path"] is not None else None,
            message=str(row["message"]),
        )
        for row in rows
    )


def _issue(check: str, path: Path | None, message: str) -> AcceptanceEvidenceIssue:
    return AcceptanceEvidenceIssue(check=check, report_path=path, message=message)


__all__ = [
    "AcceptanceEvidenceIssue",
    "AcceptanceEvidenceReport",
    "render_acceptance_evidence_report",
    "verify_acceptance_evidence",
    "write_acceptance_evidence_report",
]
