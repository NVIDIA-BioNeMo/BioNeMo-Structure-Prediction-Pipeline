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

"""Temporary Phase 1 workflow safety policies for concrete RunSpecs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bspp.orchestration.contract.runspec import (
    RunKind,
    RunSpec,
)
from bspp.orchestration.contract.runspec_validation import (
    AFDB_TOOLKIT_CONTAINER_TARGET,
    ORCHESTRATION_CONTAINER_TARGET,
    has_mount_target,
)

PARITY_TIME_LIMIT_SECONDS = 30 * 60
ACCEPTANCE_GATE_STEPS = frozenset(
    {
        "acceptance-tar-payload-parity",
        "acceptance-semantic",
        "acceptance-verify-evidence",
    }
)
SourceState = Literal["unknown", "dirty", "committed"]


@dataclass(frozen=True)
class Phase1PolicyResult:
    """One Phase 1 guardrail result."""

    code: str
    ok: bool
    message: str
    blocker: bool = True
    details: Mapping[str, object] = field(default_factory=dict)

    def to_redacted_dict(self) -> dict[str, object]:
        """Return deterministic JSON-ready policy data."""
        return {
            "code": self.code,
            "ok": self.ok,
            "message": self.message,
            "blocker": self.blocker,
            "details": dict(self.details),
        }


class Phase1PolicyError(ValueError):
    """Raised when Phase 1 policies block a RunSpec."""

    def __init__(self, failures: tuple[Phase1PolicyResult, ...]) -> None:
        self.failures = failures
        codes = ", ".join(result.code for result in failures)
        messages = "; ".join(f"{result.code}: {result.message}" for result in failures)
        super().__init__(f"Phase 1 RunSpec policies blocked execution ({codes}): {messages}")


@dataclass(frozen=True)
class RunKindPolicyContext:
    """External context for Run Kind policy checks."""

    run_kind: RunKind
    source_state: SourceState = "unknown"
    allow_dirty_source: bool = False


@dataclass(frozen=True)
class RunKindPolicyEvaluation:
    """Run Kind policy results plus runtime requirements."""

    results: tuple[Phase1PolicyResult, ...]
    requires_confirmation: bool
    requires_runtime_qualification: bool
    requires_committed_source: bool


class RunKindPolicyError(ValueError):
    """Raised when Run Kind policies block a RunSpec."""

    def __init__(self, evaluation: RunKindPolicyEvaluation) -> None:
        self.evaluation = evaluation
        failures = tuple(result for result in evaluation.results if result.blocker and not result.ok)
        self.failures = failures
        codes = ", ".join(result.code for result in failures)
        messages = "; ".join(f"{result.code}: {result.message}" for result in failures)
        super().__init__(f"Run Kind policies blocked execution ({codes}): {messages}")


def provenance_governed(run_kind: RunKind) -> bool:
    """Return whether release provenance is mandatory for a run kind."""
    return run_kind in {"canary", "production"}


def evaluate_phase1_policies(spec: RunSpec) -> tuple[Phase1PolicyResult, ...]:
    """Evaluate every removable Phase 1 policy in stable code order."""
    return (
        _allow_production_prefixes_policy(spec),
        _gcs_destination_prefix_policy(spec),
        _local_tar_self_upload_policy(spec),
        _absolute_sqsh_image_policy(spec),
        _sqsh_image_suffix_policy(spec),
        _orchestration_mount_target_policy(spec),
        _afdb_toolkit_mount_target_policy(spec),
        _parity_time_limit_policy(spec),
    )


def enforce_phase1_policies(spec: RunSpec) -> None:
    """Raise when any Phase 1 policy blocks the RunSpec."""
    failures = tuple(result for result in evaluate_phase1_policies(spec) if result.blocker and not result.ok)
    if failures:
        raise Phase1PolicyError(failures)


def evaluate_run_kind_policies(spec: RunSpec, context: RunKindPolicyContext) -> RunKindPolicyEvaluation:
    """Evaluate Run Kind policy checks for a concrete RunSpec."""
    requires_confirmation = context.run_kind == "production"
    requires_runtime_qualification = requires_confirmation
    requires_committed_source = provenance_governed(context.run_kind)
    results = (
        _run_kind_production_acceptance_policy(spec, context),
        _run_kind_source_policy(context),
        *_universal_phase1_policies(spec),
    )
    return RunKindPolicyEvaluation(
        results=results,
        requires_confirmation=requires_confirmation,
        requires_runtime_qualification=requires_runtime_qualification,
        requires_committed_source=requires_committed_source,
    )


def enforce_run_kind_policies(spec: RunSpec, context: RunKindPolicyContext) -> None:
    """Raise when any Run Kind policy blocks the RunSpec."""
    evaluation = evaluate_run_kind_policies(spec, context)
    if any(result.blocker and not result.ok for result in evaluation.results):
        raise RunKindPolicyError(evaluation)


def enforce_source_policy(context: RunKindPolicyContext) -> None:
    """Raise when the Run Kind source policy blocks the current source state."""
    result = _run_kind_source_policy(context)
    evaluation = RunKindPolicyEvaluation(
        results=(result,),
        requires_confirmation=False,
        requires_runtime_qualification=False,
        requires_committed_source=provenance_governed(context.run_kind),
    )
    if result.blocker and not result.ok:
        raise RunKindPolicyError(evaluation)


def policy_results_for(spec: RunSpec, *, source_state: SourceState = "unknown") -> tuple[Phase1PolicyResult, ...]:
    """Return legacy Phase 1 or Run Kind policy results for a RunSpec."""
    if spec.run_kind is None:
        return evaluate_phase1_policies(spec)
    context = RunKindPolicyContext(run_kind=spec.run_kind, source_state=source_state)
    return evaluate_run_kind_policies(spec, context).results


def parse_slurm_time_seconds(value: str) -> int:
    """Parse a Slurm [D-]HH:MM:SS time string into seconds."""
    day_count = 0
    clock = value
    if "-" in value:
        day_text, clock = value.split("-", 1)
        day_count = _parse_non_negative_int(day_text, field_name="days")

    parts = clock.split(":")
    if len(parts) != 3:
        msg = f"Expected Slurm time in [D-]HH:MM:SS form, got {value!r}"
        raise ValueError(msg)

    hours = _parse_non_negative_int(parts[0], field_name="hours")
    minutes = _parse_non_negative_int(parts[1], field_name="minutes")
    seconds = _parse_non_negative_int(parts[2], field_name="seconds")
    if minutes >= 60 or seconds >= 60:
        msg = f"Expected minutes and seconds below 60 in Slurm time {value!r}"
        raise ValueError(msg)
    return (day_count * 24 * 60 * 60) + (hours * 60 * 60) + (minutes * 60) + seconds


def _run_kind_production_acceptance_policy(spec: RunSpec, context: RunKindPolicyContext) -> Phase1PolicyResult:
    applies = context.run_kind == "production" and _is_local_tar_output(spec)
    missing = _missing_acceptance_gate_steps(spec) if applies else ()
    ok = not missing
    return Phase1PolicyResult(
        code="BSPP-RK-003",
        ok=ok,
        message=(
            "production local-tar runs require acceptance tar parity, semantic, and evidence gates"
            if not ok
            else "production local-tar acceptance gate policy is satisfied"
        ),
        details={
            "run_kind": context.run_kind,
            "local_tar_output": _is_local_tar_output(spec),
            "required_steps": tuple(sorted(ACCEPTANCE_GATE_STEPS)),
            "missing_steps": missing,
        },
    )


def _run_kind_source_policy(context: RunKindPolicyContext) -> Phase1PolicyResult:
    committed_source_required = context.run_kind in {"canary", "production"}
    dirty_dev_allowed = context.run_kind == "dev" and context.source_state == "dirty" and context.allow_dirty_source
    ok = context.source_state != "dirty" or dirty_dev_allowed
    if not ok and committed_source_required:
        message = "canary and production runs require committed source; source_state=dirty is blocked"
    elif not ok:
        message = "dirty source requires --allow-dirty-source for dev runs"
    else:
        message = "Run Kind source policy is satisfied"
    return Phase1PolicyResult(
        code="BSPP-RK-004",
        ok=ok,
        message=message,
        details={
            "run_kind": context.run_kind,
            "source_state": context.source_state,
            "allow_dirty_source": context.allow_dirty_source,
        },
    )


def _universal_phase1_policies(spec: RunSpec) -> tuple[Phase1PolicyResult, ...]:
    return (
        _local_tar_self_upload_policy(spec),
        _absolute_sqsh_image_policy(spec),
        _sqsh_image_suffix_policy(spec),
        _orchestration_mount_target_policy(spec),
        _afdb_toolkit_mount_target_policy(spec),
        _parity_time_limit_policy(spec),
    )


def _missing_acceptance_gate_steps(spec: RunSpec) -> tuple[str, ...]:
    enabled = _enabled_workflow_steps(spec)
    return tuple(step for step in sorted(ACCEPTANCE_GATE_STEPS) if step not in enabled)


def _enabled_workflow_steps(spec: RunSpec) -> frozenset[str]:
    if spec.workflow is None:
        return frozenset()
    return frozenset(step.name for step in spec.workflow.steps if step.run)


def _is_local_tar_output(spec: RunSpec) -> bool:
    return spec.storage.upload_mode == "tar" and spec.storage.local_tar_dir is not None


def _parse_non_negative_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        msg = f"Expected numeric {field_name} in Slurm time, got {value!r}"
        raise ValueError(msg) from exc
    if parsed < 0:
        msg = f"Expected non-negative {field_name} in Slurm time, got {value!r}"
        raise ValueError(msg)
    return parsed


def _allow_production_prefixes_policy(spec: RunSpec) -> Phase1PolicyResult:
    ok = not spec.storage.allow_production_prefixes
    return Phase1PolicyResult(
        code="BSPP-P1-001",
        ok=ok,
        message=(
            "storage.allow_production_prefixes must remain false in Phase 1"
            if not ok
            else "production-prefix override is disabled"
        ),
        details={"storage.allow_production_prefixes": spec.storage.allow_production_prefixes},
    )


def _gcs_destination_prefix_policy(spec: RunSpec) -> Phase1PolicyResult:
    ok = spec.storage.gcs_destination_prefix is None
    return Phase1PolicyResult(
        code="BSPP-P1-002",
        ok=ok,
        message=(
            "storage.gcs_destination_prefix must be null in Phase 1" if not ok else "GCS destination prefix is disabled"
        ),
        details={"storage.gcs_destination_prefix": spec.storage.gcs_destination_prefix},
    )


def _local_tar_self_upload_policy(spec: RunSpec) -> Phase1PolicyResult:
    ok = not (spec.storage.local_tar_dir is not None and spec.worker.self_upload)
    return Phase1PolicyResult(
        code="BSPP-P1-003",
        ok=ok,
        message=(
            "local-tar workflows must set worker.self_upload=false in Phase 1"
            if not ok
            else "local-tar workflow is not combined with worker self-upload"
        ),
        details={
            "storage.local_tar_dir": spec.storage.local_tar_dir,
            "worker.self_upload": spec.worker.self_upload,
        },
    )


def _absolute_sqsh_image_policy(spec: RunSpec) -> Phase1PolicyResult:
    ok = Path(spec.container.image).is_absolute()
    return Phase1PolicyResult(
        code="BSPP-P1-004",
        ok=ok,
        message=(
            "container.image must be an absolute .sqsh path in Phase 1"
            if not ok
            else "container image path is absolute"
        ),
        details={"container.image": spec.container.image},
    )


def _sqsh_image_suffix_policy(spec: RunSpec) -> Phase1PolicyResult:
    ok = spec.container.image.endswith(".sqsh")
    return Phase1PolicyResult(
        code="BSPP-P1-005",
        ok=ok,
        message=(
            "container.image must point to a .sqsh image in Phase 1" if not ok else "container image suffix is .sqsh"
        ),
        details={"container.image": spec.container.image},
    )


def _orchestration_mount_target_policy(spec: RunSpec) -> Phase1PolicyResult:
    ok = has_mount_target(spec, ORCHESTRATION_CONTAINER_TARGET)
    return Phase1PolicyResult(
        code="BSPP-P1-006",
        ok=ok,
        message=(
            f"container.mounts must include target {ORCHESTRATION_CONTAINER_TARGET}"
            if not ok
            else "orchestration container mount target is present"
        ),
        details={"expected_target": ORCHESTRATION_CONTAINER_TARGET},
    )


def _afdb_toolkit_mount_target_policy(spec: RunSpec) -> Phase1PolicyResult:
    if spec.paths.afdb_toolkit_repo is None:
        # Baked mode: toolkit is in the image, no host mount needed.
        return Phase1PolicyResult(
            code="BSPP-P1-007",
            ok=True,
            message="AFDB toolkit is image-baked; no host mount required",
            details={"expected_target": AFDB_TOOLKIT_CONTAINER_TARGET, "baked_mode": True},
        )
    ok = has_mount_target(spec, AFDB_TOOLKIT_CONTAINER_TARGET)
    return Phase1PolicyResult(
        code="BSPP-P1-007",
        ok=ok,
        message=(
            f"container.mounts must include target {AFDB_TOOLKIT_CONTAINER_TARGET}"
            if not ok
            else "AFDB toolkit container mount target is present"
        ),
        details={"expected_target": AFDB_TOOLKIT_CONTAINER_TARGET, "baked_mode": False},
    )


def _parity_time_limit_policy(spec: RunSpec) -> Phase1PolicyResult:
    resource = spec.resources.get("acceptance_tar_payload_parity")
    if resource is None:
        return Phase1PolicyResult(
            code="BSPP-P1-008",
            ok=True,
            message="acceptance tar-payload parity resource is absent",
            details={"resource_present": False, "limit_seconds": PARITY_TIME_LIMIT_SECONDS},
        )

    try:
        seconds = parse_slurm_time_seconds(resource.time)
    except ValueError as exc:
        return Phase1PolicyResult(
            code="BSPP-P1-008",
            ok=False,
            message=f"acceptance tar-payload parity time is invalid: {exc}",
            details={
                "resource_present": True,
                "time": resource.time,
                "limit_seconds": PARITY_TIME_LIMIT_SECONDS,
            },
        )

    ok = seconds <= PARITY_TIME_LIMIT_SECONDS
    return Phase1PolicyResult(
        code="BSPP-P1-008",
        ok=ok,
        message=(
            "acceptance tar-payload parity time must be at most 1800 seconds in Phase 1"
            if not ok
            else "acceptance tar-payload parity time is within Phase 1 limit"
        ),
        details={
            "resource_present": True,
            "time": resource.time,
            "seconds": seconds,
            "limit_seconds": PARITY_TIME_LIMIT_SECONDS,
        },
    )


__all__ = [
    "ACCEPTANCE_GATE_STEPS",
    "PARITY_TIME_LIMIT_SECONDS",
    "Phase1PolicyError",
    "Phase1PolicyResult",
    "RunKindPolicyContext",
    "RunKindPolicyError",
    "RunKindPolicyEvaluation",
    "SourceState",
    "enforce_phase1_policies",
    "enforce_run_kind_policies",
    "enforce_source_policy",
    "evaluate_phase1_policies",
    "evaluate_run_kind_policies",
    "parse_slurm_time_seconds",
    "policy_results_for",
]
