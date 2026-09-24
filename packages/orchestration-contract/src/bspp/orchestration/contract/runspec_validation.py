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

"""Static, file-existence-free validation for workflow RunSpecs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from bspp.orchestration.contract.runspec import RunSpec

ORCHESTRATION_CONTAINER_TARGET = Path("/workspace/bspp-orchestration")
# Historical V1/V2 authority records were authored against the pre-rename
# repository mount target; the loader must keep accepting them byte-identically
# while new materialization uses the renamed target.
LEGACY_ORCHESTRATION_CONTAINER_TARGET = Path("/workspace/afcdb-orchestration")
AFDB_TOOLKIT_CONTAINER_TARGET = Path("/workspace/AFDB-Integration-Kit")

_WORKFLOW_ORDER_RULES = (
    ("recipe", "preprocess"),
    ("preprocess", "slurm"),
    ("slurm", "analysis-finalize"),
    ("slurm", "acceptance-tar-payload-parity"),
    ("analysis-finalize", "acceptance-tar-payload-parity"),
    ("slurm", "acceptance-semantic"),
    ("analysis-finalize", "acceptance-semantic"),
    ("slurm", "acceptance-verify-evidence"),
    ("analysis-finalize", "acceptance-verify-evidence"),
    ("acceptance-tar-payload-parity", "acceptance-verify-evidence"),
    ("acceptance-semantic", "acceptance-verify-evidence"),
)


@dataclass(frozen=True)
class StaticValidationIssue:
    """One static workflow validation issue."""

    code: str
    message: str
    blocker: bool = True
    details: Mapping[str, object] = field(default_factory=dict)

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready issue data."""
        return {
            "code": self.code,
            "message": self.message,
            "blocker": self.blocker,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class StaticValidationResult:
    """Aggregate static workflow validation result."""

    issues: tuple[StaticValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Return true when no blocking static issue exists."""
        return not self.blockers

    @property
    def blockers(self) -> tuple[str, ...]:
        """Return blocking issue messages with stable codes."""
        return tuple(f"{issue.code}: {issue.message}" for issue in self.issues if issue.blocker)

    def raise_if_invalid(self) -> None:
        """Raise if static validation found blockers."""
        if not self.ok:
            raise StaticValidationError(self)

    def to_redacted_dict(self) -> dict[str, object]:
        """Return JSON-ready validation data."""
        return {
            "ok": self.ok,
            "blockers": list(self.blockers),
            "issues": [issue.to_redacted_dict() for issue in self.issues],
        }


class StaticValidationError(ValueError):
    """Raised when static workflow validation blocks a RunSpec."""

    def __init__(self, result: StaticValidationResult) -> None:
        self.result = result
        super().__init__("RunSpec static validation failed: " + "; ".join(result.blockers))


def validate_active_workflow_static(spec: RunSpec) -> StaticValidationResult:
    """Validate active workflow ordering and code mounts without probing files."""
    if spec.workflow is None:
        return StaticValidationResult()

    issues: list[StaticValidationIssue] = []
    issues.extend(_workflow_order_issues(spec))
    issues.extend(_code_mount_issues(spec))
    issues.extend(_acceptance_config_issues(spec))
    return StaticValidationResult(tuple(issues))


def has_mount_target(spec: RunSpec, target: Path) -> bool:
    """Return true when a container mount uses *target*."""
    return any(mount.target == target for mount in spec.container.mounts)


def has_source_to_target_mount(spec: RunSpec, *, source: Path, target: Path) -> bool:
    """Return true when a container mount maps *source* exactly to *target*."""
    return any(mount.source == source and mount.target == target for mount in spec.container.mounts)


def _workflow_order_issues(spec: RunSpec) -> tuple[StaticValidationIssue, ...]:
    if spec.workflow is None:
        return ()

    positions = {step.name: index for index, step in enumerate(spec.workflow.steps)}
    issues: list[StaticValidationIssue] = []
    for prerequisite, dependent in _WORKFLOW_ORDER_RULES:
        prerequisite_position = positions.get(prerequisite)
        dependent_position = positions.get(dependent)
        if prerequisite_position is None or dependent_position is None:
            continue
        if prerequisite_position > dependent_position:
            issues.append(
                StaticValidationIssue(
                    code="BSPP-STATIC-001",
                    message=(f"workflow step {prerequisite!r} must appear before {dependent!r} when both are present"),
                    details={
                        "prerequisite": prerequisite,
                        "dependent": dependent,
                        "prerequisite_position": prerequisite_position,
                        "dependent_position": dependent_position,
                    },
                )
            )
    return tuple(issues)


def _code_mount_issues(spec: RunSpec) -> tuple[StaticValidationIssue, ...]:
    issues: list[StaticValidationIssue] = []
    if not has_source_to_target_mount(
        spec,
        source=spec.paths.orchestration_repo,
        target=ORCHESTRATION_CONTAINER_TARGET,
    ) and not has_source_to_target_mount(
        spec,
        source=spec.paths.orchestration_repo,
        target=LEGACY_ORCHESTRATION_CONTAINER_TARGET,
    ):
        issues.append(
            StaticValidationIssue(
                code="BSPP-STATIC-002",
                message=(
                    f"container.mounts must map paths.orchestration_repo to {ORCHESTRATION_CONTAINER_TARGET} "
                    f"(or legacy {LEGACY_ORCHESTRATION_CONTAINER_TARGET})"
                ),
                details={
                    "expected_source": spec.paths.orchestration_repo,
                    "expected_target": ORCHESTRATION_CONTAINER_TARGET,
                    "legacy_target": LEGACY_ORCHESTRATION_CONTAINER_TARGET,
                },
            )
        )

    if spec.paths.afdb_toolkit_repo is None:
        # Baked mode: no toolkit path, no mount required. Pass silently.
        return tuple(issues)

    if not has_source_to_target_mount(
        spec,
        source=spec.paths.afdb_toolkit_repo,
        target=AFDB_TOOLKIT_CONTAINER_TARGET,
    ):
        issues.append(
            StaticValidationIssue(
                code="BSPP-STATIC-003",
                message=(f"container.mounts must map paths.afdb_toolkit_repo to {AFDB_TOOLKIT_CONTAINER_TARGET}"),
                details={
                    "expected_source": spec.paths.afdb_toolkit_repo,
                    "expected_target": AFDB_TOOLKIT_CONTAINER_TARGET,
                },
            )
        )
    return tuple(issues)


def _acceptance_config_issues(spec: RunSpec) -> tuple[StaticValidationIssue, ...]:
    if spec.workflow is None:
        return ()

    enabled_steps = {step.name for step in spec.workflow.steps if step.run}
    issues: list[StaticValidationIssue] = []
    if "acceptance-tar-payload-parity" in enabled_steps:
        missing = _missing_acceptance_requirements(spec, "acceptance_tar_payload_parity")
        if missing:
            issues.append(
                StaticValidationIssue(
                    code="BSPP-STATIC-004",
                    message=(
                        "enabled acceptance-tar-payload-parity requires top-level acceptance, "
                        "acceptance.baseline_output_dir, and resources.acceptance_tar_payload_parity"
                    ),
                    details={"missing": missing},
                )
            )

    if "acceptance-semantic" in enabled_steps:
        missing = _missing_acceptance_requirements(spec, "acceptance_semantic")
        if missing:
            issues.append(
                StaticValidationIssue(
                    code="BSPP-STATIC-005",
                    message=(
                        "enabled acceptance-semantic requires top-level acceptance, "
                        "acceptance.baseline_output_dir, and resources.acceptance_semantic"
                    ),
                    details={"missing": missing},
                )
            )

    if "acceptance-verify-evidence" in enabled_steps and not (
        "acceptance-tar-payload-parity" in enabled_steps or "acceptance-semantic" in enabled_steps
    ):
        issues.append(
            StaticValidationIssue(
                code="BSPP-STATIC-006",
                message="enabled acceptance-verify-evidence requires at least one enabled acceptance comparator step",
                details={"enabled_acceptance_comparators": []},
            )
        )

    return tuple(issues)


def _missing_acceptance_requirements(spec: RunSpec, resource_key: str) -> tuple[str, ...]:
    missing: list[str] = []
    if spec.acceptance is None:
        missing.extend(["acceptance", "acceptance.baseline_output_dir"])
    elif spec.acceptance.baseline_output_dir is None:
        missing.append("acceptance.baseline_output_dir")
    if resource_key not in spec.resources:
        missing.append(f"resources.{resource_key}")
    return tuple(missing)


__all__ = [
    "AFDB_TOOLKIT_CONTAINER_TARGET",
    "ORCHESTRATION_CONTAINER_TARGET",
    "StaticValidationError",
    "StaticValidationIssue",
    "StaticValidationResult",
    "has_mount_target",
    "has_source_to_target_mount",
    "validate_active_workflow_static",
]
