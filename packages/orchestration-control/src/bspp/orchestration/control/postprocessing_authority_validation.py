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

"""Cross-version validation helpers for persisted postprocessing authority."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

import yaml

from bspp.orchestration.contract.postprocessing_event import (
    PostprocessingPhaseEvent,
    postprocessing_phase_event_from_mapping,
)
from bspp.orchestration.contract.postprocessing_execution import (
    QualifiedPostprocessingRuntimeSelection,
)
from bspp.orchestration.contract.postprocessing_plan import PostprocessingPhasePlan
from bspp.orchestration.contract.runspec import RunSpec, runspec_from_mapping
from bspp.orchestration.contract.runspec_validation import validate_active_workflow_static
from bspp.orchestration.control.plan import render_runspec_yaml
from bspp.orchestration.control.postprocessing_authority_store import (
    read_canonical_json,
    required_string,
    verify_bytes,
)
from bspp.orchestration.control.postprocessing_phase_lifecycle import parse_postprocessing_timestamp
from bspp.orchestration.control.postprocessing_runtime_qualification import replay_postprocessing_runtime

_EVENT_NAME = re.compile(r"([0-9]{6})-([a-z][a-z0-9-]*)\.json")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def replay_qualified_postprocessing_runtime(
    document: bytes,
    *,
    attempt_id: str,
    profile_name: str,
) -> QualifiedPostprocessingRuntimeSelection:
    """Rebuild one frozen Runtime selection without materialization or ambient state."""
    replayed_profile, selection = replay_postprocessing_runtime(document, attempt_id=attempt_id)
    if replayed_profile != profile_name:
        raise ValueError("postprocessing Runtime Qualification profile differs from frozen Attempt authority")
    return selection


def validate_postprocessing_phase_run_mapping(
    payload: Mapping[str, object],
    *,
    plan: PostprocessingPhasePlan,
    expected_phase_run_id: str,
) -> None:
    allowed = {
        "schema_version",
        "phase_kind",
        "phase_run_id",
        "phase_plan_location",
        "phase_plan_digest",
        "created_at",
        "current_attempt_id",
        "attempts",
        "status",
        "sealed",
    }
    if set(payload) != allowed or payload.get("schema_version") != 1 or payload.get("phase_kind") != "postprocessing":
        raise ValueError("stored postprocessing Phase Run has an invalid strict envelope")
    if payload.get("phase_run_id") != expected_phase_run_id or payload.get("phase_plan_location") != "phase-plan.json":
        raise ValueError("stored postprocessing Phase Run identity is invalid")
    if payload.get("phase_plan_digest") != plan.digest:
        raise ValueError("stored postprocessing Phase Run plan digest differs")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 1 or not isinstance(attempts[0], Mapping):
        raise ValueError("stored postprocessing Phase Run must declare exactly its initial Attempt")
    current = required_string(payload, "current_attempt_id")
    if current != "attempt-0001":
        raise ValueError("stored postprocessing Phase Run current Attempt snapshot must be attempt-0001")
    created_at = required_string(payload, "created_at")
    parse_postprocessing_timestamp(created_at)
    if payload.get("status") != "materialized" or payload.get("sealed") is not False:
        raise ValueError("stored postprocessing Phase Run must retain its immutable materialized snapshot")
    for ordinal, item in enumerate(attempts, start=1):
        attempt_fields = {
            "schema_version",
            "attempt_id",
            "ordinal",
            "phase_runspec_location",
            "phase_runspec_digest",
            "created_at",
            "status",
        }
        if set(item) != attempt_fields:
            raise ValueError("stored postprocessing Attempt has an invalid strict envelope")
        if item.get("schema_version") != 1 or item.get("attempt_id") != f"attempt-{ordinal:04d}":
            raise ValueError("stored postprocessing Attempt identities must be contiguous")
        if (
            item.get("ordinal") != ordinal
            or item.get("created_at") != created_at
            or item.get("status") != "materialized"
        ):
            raise ValueError("stored postprocessing initial Attempt snapshot is invalid")
        if item.get("phase_runspec_location") != f"attempts/attempt-{ordinal:04d}/phase-runspec.json":
            raise ValueError("stored postprocessing Attempt RunSpec location is invalid")
        digest = item.get("phase_runspec_digest")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError("stored postprocessing Attempt RunSpec digest is invalid")


def read_postprocessing_events(authority_path: Path) -> tuple[PostprocessingPhaseEvent, ...]:
    event_paths = sorted((authority_path / "events").glob("*.json"))
    if not event_paths:
        raise ValueError("postprocessing authority has no materialization event")
    events: list[PostprocessingPhaseEvent] = []
    for expected_sequence, path in enumerate(event_paths, start=1):
        match = _EVENT_NAME.fullmatch(path.name)
        if match is None or int(match.group(1)) != expected_sequence:
            raise ValueError("postprocessing authority events must be contiguous and canonically named")
        raw_event = read_canonical_json(path)
        if raw_event.get("sequence") != expected_sequence or raw_event.get("event_type") != match.group(2):
            raise ValueError("postprocessing event filename and envelope differ")
        if raw_event.get("schema_version") != 1 or raw_event.get("phase_kind") != "postprocessing":
            raise ValueError("postprocessing event has an invalid family/version discriminator")
        events.append(postprocessing_phase_event_from_mapping(raw_event))
    if events[0].event_type != "phase-materialized":
        raise ValueError("postprocessing authority must start with phase-materialized")
    return tuple(events)


def legacy_runspec_from_bytes(payload: bytes, *, source_path: Path, expected_sha256: str) -> RunSpec:
    mapping = yaml.safe_load(payload)
    if not isinstance(mapping, Mapping):
        raise TypeError("stored legacy RunSpec projection must be a YAML mapping")
    spec = runspec_from_mapping(mapping, source_path=source_path, source_hash=expected_sha256)
    validate_active_workflow_static(spec).raise_if_invalid()
    if render_runspec_yaml(mapping).encode() != payload:
        raise ValueError("stored legacy RunSpec projection bytes are not the once-rendered canonical YAML")
    return spec


def verified_regular_bytes(path: Path, *, expected_sha256: str, expected_size: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    payload = path.read_bytes()
    verify_bytes(payload, expected_sha256=expected_sha256, expected_size=expected_size, label=label)
    return payload


def verify_optional_projection(path: Path, expected: bytes) -> bool:
    if not os.path.lexists(path):
        return False
    if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
        raise ValueError(f"existing postprocessing Retry projection differs from embedded authority: {path}")
    return True


def projection_bytes_with_initial_fallback(
    path: Path,
    *,
    initial_path: Path,
    expected_sha256: str,
    expected_size: int,
    label: str,
) -> tuple[bytes, bool]:
    if os.path.lexists(path):
        return (
            verified_regular_bytes(
                path,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
                label=label,
            ),
            True,
        )
    return (
        verified_regular_bytes(
            initial_path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            label=f"initial fallback for {label}",
        ),
        False,
    )


__all__ = [
    "legacy_runspec_from_bytes",
    "projection_bytes_with_initial_fallback",
    "read_postprocessing_events",
    "replay_qualified_postprocessing_runtime",
    "validate_postprocessing_phase_run_mapping",
    "verified_regular_bytes",
    "verify_optional_projection",
]
