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

"""Workflow-only RunSpec preflight reports."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path

from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.contract.runspec_policies import Phase1PolicyResult, policy_results_for
from bspp.orchestration.contract.runspec_validation import StaticValidationResult, validate_active_workflow_static
from bspp.orchestration.runtime.inputs.reports import report_to_json, write_json_report, write_text_summary


@dataclass(frozen=True)
class WorkflowStepSummary:
    """One requested workflow step as recorded in preflight evidence."""

    name: str
    run: bool
    mode: str | None
    job_id: str | None
    array_range: str | None
    rendered_script: Path | None

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-ready step data."""
        return {
            "name": self.name,
            "run": self.run,
            "mode": self.mode,
            "job_id": self.job_id,
            "array_range": self.array_range,
            "rendered_script": self.rendered_script,
        }


@dataclass(frozen=True)
class WorkflowSummary:
    """Ordered enabled/skipped workflow step evidence."""

    steps: tuple[WorkflowStepSummary, ...]

    @property
    def enabled_steps(self) -> tuple[str, ...]:
        """Return requested steps with run=true in YAML order."""
        return tuple(step.name for step in self.steps if step.run)

    @property
    def skipped_steps(self) -> tuple[str, ...]:
        """Return requested steps with run=false in YAML order."""
        return tuple(step.name for step in self.steps if not step.run)

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-ready workflow summary data."""
        return {
            "total_steps": len(self.steps),
            "enabled_steps": list(self.enabled_steps),
            "skipped_steps": list(self.skipped_steps),
            "steps": [step.to_redacted_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class RunSpecPreflightReport:
    """Workflow-only preflight report for one concrete RunSpec."""

    schema_version: int
    source_runspec: Path | None
    source_hash: str | None
    dataset: str
    run_id: str
    cluster_name: str
    hostname: str
    hostname_matches_cluster: bool
    workflow_summary: WorkflowSummary
    static_validation: StaticValidationResult
    phase1_policies: tuple[Phase1PolicyResult, ...]

    @property
    def ready(self) -> bool:
        """Return true when static validation and Phase 1 policies have no blockers."""
        return not self.blockers

    @property
    def blockers(self) -> tuple[str, ...]:
        """Return blocking messages; hostname mismatch is evidence-only."""
        policy_blockers = tuple(
            f"{result.code}: {result.message}" for result in self.phase1_policies if result.blocker and not result.ok
        )
        return self.static_validation.blockers + policy_blockers

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-ready preflight evidence."""
        return {
            "schema_version": self.schema_version,
            "source_runspec": self.source_runspec,
            "source_hash": self.source_hash,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "cluster_name": self.cluster_name,
            "hostname": self.hostname,
            "hostname_matches_cluster": self.hostname_matches_cluster,
            "workflow_summary": self.workflow_summary.to_redacted_dict(),
            "static_validation": self.static_validation.to_redacted_dict(),
            "phase1_policies": [result.to_redacted_dict() for result in self.phase1_policies],
            "ready": self.ready,
            "blockers": list(self.blockers),
        }


class ActiveWorkflowSafetyError(ValueError):
    """Raised when static validation or Phase 1 policies block execution."""

    def __init__(
        self,
        *,
        static_validation: StaticValidationResult,
        phase1_policies: tuple[Phase1PolicyResult, ...],
    ) -> None:
        self.static_validation = static_validation
        self.phase1_policies = phase1_policies
        policy_blockers = tuple(
            f"{result.code}: {result.message}" for result in phase1_policies if result.blocker and not result.ok
        )
        blockers = static_validation.blockers + policy_blockers
        super().__init__("RunSpec workflow safety blocked execution: " + "; ".join(blockers))


def build_runspec_preflight_report(
    spec: RunSpec,
    *,
    hostname: str | None = None,
) -> RunSpecPreflightReport:
    """Build workflow preflight evidence without probing local files."""
    if spec.workflow is None or spec.submission is None:
        msg = "runspec preflight requires an active workflow RunSpec with workflow and submission sections"
        raise ValueError(msg)

    actual_hostname = hostname if hostname is not None else socket.gethostname()
    return RunSpecPreflightReport(
        schema_version=1,
        source_runspec=spec.source_path,
        source_hash=spec.source_hash,
        dataset=spec.dataset.name,
        run_id=spec.dataset.run_id,
        cluster_name=spec.cluster.name,
        hostname=actual_hostname,
        hostname_matches_cluster=_hostname_matches_cluster(actual_hostname, spec.cluster.name),
        workflow_summary=_workflow_summary(spec),
        static_validation=validate_active_workflow_static(spec),
        phase1_policies=policy_results_for(spec),
    )


def enforce_active_workflow_safety(spec: RunSpec) -> None:
    """Run static validation and Phase 1 policies before workflow execution."""
    static_validation = validate_active_workflow_static(spec)
    phase1_policies = policy_results_for(spec)
    policy_blocked = any(result.blocker and not result.ok for result in phase1_policies)
    if not static_validation.ok or policy_blocked:
        raise ActiveWorkflowSafetyError(
            static_validation=static_validation,
            phase1_policies=phase1_policies,
        )


def render_runspec_preflight_report(report: RunSpecPreflightReport) -> str:
    """Render deterministic JSON workflow preflight evidence."""
    return report_to_json(report)


def write_runspec_preflight_reports(report: RunSpecPreflightReport, evidence_dir: Path) -> tuple[Path, Path]:
    """Write workflow preflight JSON and text reports under submission evidence."""
    report_dir = evidence_dir / "preflight"
    json_path = write_json_report(report, report_dir / "preflight_report.json")
    text_path = write_text_summary(report, report_dir / "preflight_report.txt")
    return json_path, text_path


def _workflow_summary(spec: RunSpec) -> WorkflowSummary:
    if spec.workflow is None:
        return WorkflowSummary(())
    return WorkflowSummary(
        tuple(
            WorkflowStepSummary(
                name=step.name,
                run=step.run,
                mode=step.mode,
                job_id=step.job_id,
                array_range=step.array_range,
                rendered_script=step.rendered_script,
            )
            for step in spec.workflow.steps
        )
    )


def _hostname_matches_cluster(hostname: str, cluster_name: str) -> bool:
    normalized_hostname = hostname.casefold().replace("_", "-")
    normalized_cluster = cluster_name.casefold().replace("_", "-")
    return normalized_cluster in normalized_hostname


__all__ = [
    "ActiveWorkflowSafetyError",
    "RunSpecPreflightReport",
    "WorkflowStepSummary",
    "WorkflowSummary",
    "build_runspec_preflight_report",
    "enforce_active_workflow_safety",
    "render_runspec_preflight_report",
    "write_runspec_preflight_reports",
]
