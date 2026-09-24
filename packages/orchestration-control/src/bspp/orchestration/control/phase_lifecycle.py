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

"""Shared classification for replayed Phase lifecycle operation gates."""

from __future__ import annotations

from typing import Literal

from bspp.orchestration.contract.phase_state import PhaseRunLifecycleView

PhaseLifecycleClassification = Literal["active", "failed", "cancelling", "cancelled", "accepted"]


def classify_phase_lifecycle(lifecycle: PhaseRunLifecycleView) -> PhaseLifecycleClassification:
    """Classify one already-strict replay lifecycle without permissive fallbacks."""
    if (
        lifecycle.attempt_status == "materialized"
        and lifecycle.run_status == "materialized"
        and not lifecycle.sealed
        and lifecycle.phase_receipt_id is None
    ):
        return "active"
    if (
        lifecycle.attempt_status == "failed"
        and lifecycle.run_status == "failed"
        and not lifecycle.sealed
        and lifecycle.phase_receipt_id is None
    ):
        return "failed"
    if (
        lifecycle.attempt_status == "cancelling"
        and lifecycle.run_status == "cancelling"
        and not lifecycle.sealed
        and lifecycle.phase_receipt_id is None
    ):
        return "cancelling"
    if (
        lifecycle.attempt_status == "cancelled"
        and lifecycle.run_status == "cancelled"
        and not lifecycle.sealed
        and lifecycle.phase_receipt_id is None
    ):
        return "cancelled"
    if (
        lifecycle.attempt_status == "succeeded"
        and lifecycle.run_status == "accepted"
        and lifecycle.sealed
        and lifecycle.phase_receipt_id is not None
    ):
        return "accepted"
    raise ValueError("Phase lifecycle is not a recognized coherent state")


def require_active_phase_attempt(lifecycle: PhaseRunLifecycleView, *, operation: str) -> None:
    """Require the current unsealed attempt for a mutating active operation."""
    classification = classify_phase_lifecycle(lifecycle)
    if classification != "active":
        raise ValueError(f"{operation} cannot cross a {classification} Phase Attempt boundary")


def require_finalizable_phase_attempt(lifecycle: PhaseRunLifecycleView) -> None:
    """Permit active finalization and accepted idempotent comparison only."""
    classification = classify_phase_lifecycle(lifecycle)
    if classification not in {"active", "accepted"}:
        raise ValueError(f"Phase Finalization cannot cross a {classification} Phase Attempt boundary")


def require_cancellable_phase_attempt(
    lifecycle: PhaseRunLifecycleView,
) -> Literal["active", "cancelling", "cancelled"]:
    """Permit a new, continuing, or idempotently completed cancellation."""
    classification = classify_phase_lifecycle(lifecycle)
    if classification == "active":
        return "active"
    if classification == "cancelling":
        return "cancelling"
    if classification == "cancelled":
        return "cancelled"
    raise ValueError(f"Phase Cancellation cannot cross a {classification} Phase Attempt boundary")


def require_retryable_phase_attempt(
    lifecycle: PhaseRunLifecycleView,
) -> Literal["failed", "cancelled"]:
    """Permit Retry only after a durable failed or fully cancelled Attempt."""
    classification = classify_phase_lifecycle(lifecycle)
    if classification == "failed":
        return "failed"
    if classification == "cancelled":
        return "cancelled"
    raise ValueError(f"Phase Retry cannot cross a {classification} Phase Attempt boundary")


__all__ = [
    "PhaseLifecycleClassification",
    "classify_phase_lifecycle",
    "require_active_phase_attempt",
    "require_cancellable_phase_attempt",
    "require_finalizable_phase_attempt",
    "require_retryable_phase_attempt",
]
