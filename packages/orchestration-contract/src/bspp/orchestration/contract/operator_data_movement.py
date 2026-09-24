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

"""Operator-initiated data-movement resolution and planning contracts.

This additive contract module owns the **decision and planning** layer for
operator-initiated artifact transfers (publish, retry, recovery). It contains
no runtime dependencies — only contract types and helpers. The runtime
package delegates the pure policy→decision mapping to
:func:`resolve_seam_transfer_decision` and the execution tool dispatch to
:func:`resolve_execution_tool`.

The control plane calls :func:`build_plan_referenced_transfer_plan` and
:func:`build_manual_transfer_plan` to produce an :class:`OperatorTransferPlan`
without importing the runtime package.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import yaml

from bspp.orchestration.contract.phase import (
    PhaseSeamTransportPolicy,
    canonical_mapping_digest,
    folding_phase_plan_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
)
from bspp.orchestration.contract.versioning import (
    CURRENT_CONTRACT_SCHEMA_VERSION,
    validate_schema_version,
)

_S3_PREFIX = re.compile(r"^s3://[^/].*$")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z")

_SOURCE_KINDS = ("verified-local-bundled", "verified-remote-bundled", "explicit-path")
_MODES = ("plan-referenced", "manual")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_s3_prefix(prefix: str) -> str:
    """Validate and normalize an ``s3://`` object-key prefix.

    Returns the prefix with a trailing slash stripped. Raises ``ValueError``
    on an invalid prefix (empty bucket, missing ``s3://`` scheme, etc.).
    """
    if not prefix.startswith("s3://"):
        raise ValueError(f"prefix must start with s3://: {prefix!r}")
    stripped = prefix.rstrip("/")
    if _S3_PREFIX.fullmatch(stripped) is None:
        raise ValueError(f"prefix must be a non-empty s3:// object key prefix (s3://<bucket>[/<key>...]): {prefix!r}")
    return stripped


def _validate_s3_object_key(destination: str) -> None:
    """Validate that ``destination`` is a full S3 object key, not a directory prefix.

    Rejects empty-bucket and keyless/directory destinations:
    ``s3://``, ``s3:///``, ``s3://bucket`` (no key), ``s3://bucket/`` (trailing slash),
    and ``s3://bucket/prefix/`` (directory prefix).
    Requires a non-empty object-key component after the bucket.
    """
    _validate_s3_prefix(destination)
    # A trailing slash marks a directory prefix, never a full object key. This
    # must be checked on the ORIGINAL destination before any normalization,
    # otherwise ``s3://bucket/prefix/`` would be stripped to
    # ``s3://bucket/prefix`` and accepted as a key.
    if destination != destination.rstrip("/"):
        raise ValueError(
            f"OperatorTransferItem destination must be a full object key with no trailing slash "
            f"(not a directory prefix): {destination!r}"
        )
    after_scheme = destination[len("s3://") :]
    if "/" not in after_scheme:
        raise ValueError(
            f"OperatorTransferItem destination must be a full object key with a non-empty key "
            f"component after the bucket (not a directory prefix): {destination!r}"
        )
    key = after_scheme.rsplit("/", 1)[-1]
    if not key:
        raise ValueError(
            f"OperatorTransferItem destination must be a full object key with a non-empty key "
            f"component after the bucket (not a directory prefix): {destination!r}"
        )


def _validate_non_negative_int(value: object, name: str) -> int:
    """Reject non-int, bool, and negative values."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    """Strict schema-version validation that rejects ``None``/missing."""
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        raise ValueError(f"{record_name} schema_version must be declared explicitly")


def _validate_timestamp(value: str, record_name: str) -> None:
    """Validate an RFC 3339 UTC timestamp.

    Uses the stricter helper from ``contract.phase``: regex check followed by
    ``datetime.fromisoformat`` parsing and UTC-offset verification, so that
    impossible dates (e.g. ``2026-13-99T99:99:99Z``) are rejected.
    """
    if _TIMESTAMP.fullmatch(value) is None:
        msg = f"{record_name} must be an explicit UTC timestamp"
        raise ValueError(msg)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = f"{record_name} must be a valid UTC timestamp"
        raise ValueError(msg) from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        msg = f"{record_name} must be an explicit UTC timestamp"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# SeamTransferDecision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeamTransferDecision:
    """Contract-level twin of the runtime's ``SeamTransferResolution`` without the transfer callable."""

    policy: PhaseSeamTransportPolicy
    object_storage_leg: bool
    tool: str | None
    note: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "SeamTransferDecision")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "object_storage_leg": self.object_storage_leg,
            "tool": self.tool,
            "note": self.note,
        }


def seam_transfer_decision_from_mapping(payload: Mapping[str, object]) -> SeamTransferDecision:
    """Deserialize a :class:`SeamTransferDecision` from a mapping."""
    _reject_unknown_fields(
        payload,
        {"schema_version", "policy", "object_storage_leg", "tool", "note"},
        "SeamTransferDecision",
    )
    schema_version = _required_int(payload, "schema_version")
    _validate_direct_schema_version(schema_version, "SeamTransferDecision")
    policy = _required_str(payload, "policy")
    if policy not in ("publish-to-s3", "local"):
        raise ValueError(f"SeamTransferDecision unsupported policy: {policy!r}")
    object_storage_leg = payload.get("object_storage_leg")
    if not isinstance(object_storage_leg, bool):
        raise ValueError("SeamTransferDecision object_storage_leg must be a bool")
    tool = payload.get("tool")
    if tool is not None and (not isinstance(tool, str) or not tool):
        raise ValueError("SeamTransferDecision tool must be null or a non-empty string")
    note = _required_str(payload, "note")
    return SeamTransferDecision(
        schema_version=schema_version,
        policy=cast("PhaseSeamTransportPolicy", policy),
        object_storage_leg=object_storage_leg,
        tool=tool,
        note=note,
    )


def resolve_seam_transfer_decision(*, policy: PhaseSeamTransportPolicy) -> SeamTransferDecision:
    """Pure policy → decision mapping.

    ``local`` → no object-storage leg, no tool.
    ``publish-to-s3`` → object-storage leg, tool ``"s5cmd"``.
    Raises ``ValueError`` on unknown policy.
    """
    if policy == "local":
        return SeamTransferDecision(
            policy=policy,
            object_storage_leg=False,
            tool=None,
            note="local pass-through (no object-storage leg)",
        )
    if policy == "publish-to-s3":
        return SeamTransferDecision(
            policy=policy,
            object_storage_leg=True,
            tool="s5cmd",
            note="s5cmd cp against the S3 endpoint",
        )
    raise ValueError(f"unsupported phase seam transport policy: {policy!r}")


# ---------------------------------------------------------------------------
# OperatorTransferItem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorTransferItem:
    """One planned transfer."""

    source: str
    destination: str
    size_bytes: int
    sha256: str
    artifact_set_id: str | None
    artifact_location_id: str | None
    source_kind: str
    description: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "OperatorTransferItem")
        _validate_non_negative_int(self.size_bytes, "OperatorTransferItem size_bytes")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("OperatorTransferItem sha256 must be 64 lowercase hex characters")
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(f"OperatorTransferItem unsupported source_kind: {self.source_kind!r}")
        if not self.source:
            raise ValueError("OperatorTransferItem source must be non-empty")
        if self.source == self.destination:
            raise ValueError("OperatorTransferItem source must not equal destination (self-copy rejected)")
        if not self.destination.startswith("s3://"):
            raise ValueError("OperatorTransferItem destination must start with s3://")
        # Destination must be a full object key: validate bucket non-empty and
        # require a non-empty object-key component after the bucket (no trailing
        # slash / directory prefix).
        _validate_s3_object_key(self.destination)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "destination": self.destination,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "artifact_set_id": self.artifact_set_id,
            "artifact_location_id": self.artifact_location_id,
            "source_kind": self.source_kind,
            "description": self.description,
        }


def operator_transfer_item_from_mapping(payload: Mapping[str, object]) -> OperatorTransferItem:
    """Deserialize an :class:`OperatorTransferItem` from a mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "source",
            "destination",
            "size_bytes",
            "sha256",
            "artifact_set_id",
            "artifact_location_id",
            "source_kind",
            "description",
        },
        "OperatorTransferItem",
    )
    schema_version = _required_int(payload, "schema_version")
    _validate_direct_schema_version(schema_version, "OperatorTransferItem")
    source = _required_str(payload, "source")
    destination = _required_str(payload, "destination")
    size_bytes = _required_int(payload, "size_bytes")
    sha256 = _required_str(payload, "sha256")
    artifact_set_id = _optional_str(payload, "artifact_set_id")
    artifact_location_id = _optional_str(payload, "artifact_location_id")
    source_kind = _required_str(payload, "source_kind")
    description = _required_str(payload, "description")
    return OperatorTransferItem(
        schema_version=schema_version,
        source=source,
        destination=destination,
        size_bytes=size_bytes,
        sha256=sha256,
        artifact_set_id=artifact_set_id,
        artifact_location_id=artifact_location_id,
        source_kind=source_kind,
        description=description,
    )


# ---------------------------------------------------------------------------
# OperatorTransferPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorTransferPlan:
    """The full operator transfer plan."""

    mode: str
    items: tuple[OperatorTransferItem, ...]
    decision: SeamTransferDecision
    operator_initiated: bool
    authority_reference: str | None
    authority_digest: str | None
    override_destination_prefix: str | None
    dry_run: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "OperatorTransferPlan")
        if self.mode not in _MODES:
            raise ValueError(f"OperatorTransferPlan unsupported mode: {self.mode!r}")
        if not self.items:
            raise ValueError("OperatorTransferPlan items must be non-empty")
        if self.operator_initiated is not True:
            raise ValueError("OperatorTransferPlan operator_initiated must be True")
        if self.mode == "plan-referenced":
            if not self.authority_reference:
                raise ValueError("OperatorTransferPlan authority_reference must be non-empty in plan-referenced mode")
            if self.authority_digest is None or _SHA256.fullmatch(self.authority_digest) is None:
                raise ValueError(
                    "OperatorTransferPlan authority_digest must be a content digest in plan-referenced mode"
                )
        else:
            if self.authority_reference is not None:
                raise ValueError("OperatorTransferPlan authority_reference must be None in manual mode")
            if self.authority_digest is not None:
                raise ValueError("OperatorTransferPlan authority_digest must be None in manual mode")
        if self.override_destination_prefix is not None:
            _validate_s3_prefix(self.override_destination_prefix)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "items": [item.to_mapping() for item in self.items],
            "decision": self.decision.to_mapping(),
            "operator_initiated": self.operator_initiated,
            "authority_reference": self.authority_reference,
            "authority_digest": self.authority_digest,
            "override_destination_prefix": self.override_destination_prefix,
            "dry_run": self.dry_run,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, indent=2)


def operator_transfer_plan_from_mapping(payload: Mapping[str, object]) -> OperatorTransferPlan:
    """Deserialize an :class:`OperatorTransferPlan` from a mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "mode",
            "items",
            "decision",
            "operator_initiated",
            "authority_reference",
            "authority_digest",
            "override_destination_prefix",
            "dry_run",
        },
        "OperatorTransferPlan",
    )
    schema_version = _required_int(payload, "schema_version")
    _validate_direct_schema_version(schema_version, "OperatorTransferPlan")
    mode = _required_str(payload, "mode")
    items_raw = payload.get("items")
    if not isinstance(items_raw, list | tuple) or not items_raw:
        raise ValueError("OperatorTransferPlan items must be a non-empty list")
    items = tuple(operator_transfer_item_from_mapping(item) for item in items_raw if isinstance(item, Mapping))
    if len(items) != len(items_raw):
        raise ValueError("OperatorTransferPlan items must all be mappings")
    decision = seam_transfer_decision_from_mapping(_required_mapping(payload, "decision"))
    operator_initiated = payload.get("operator_initiated")
    if not isinstance(operator_initiated, bool):
        raise ValueError("OperatorTransferPlan operator_initiated must be a bool")
    authority_reference = _optional_str(payload, "authority_reference")
    authority_digest = _optional_str(payload, "authority_digest")
    override_destination_prefix = _optional_str(payload, "override_destination_prefix")
    dry_run = payload.get("dry_run")
    if not isinstance(dry_run, bool):
        raise ValueError("OperatorTransferPlan dry_run must be a bool")
    return OperatorTransferPlan(
        schema_version=schema_version,
        mode=mode,
        items=items,
        decision=decision,
        operator_initiated=operator_initiated,
        authority_reference=authority_reference,
        authority_digest=authority_digest,
        override_destination_prefix=override_destination_prefix,
        dry_run=dry_run,
    )


def operator_transfer_plan_from_json(text: str) -> OperatorTransferPlan:
    """Deserialize an :class:`OperatorTransferPlan` from a JSON string."""
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("OperatorTransferPlan JSON must be a mapping")
    return operator_transfer_plan_from_mapping(payload)


# ---------------------------------------------------------------------------
# OperatorTransferEvidence
# ---------------------------------------------------------------------------


def operator_transfer_evidence_id(payload: Mapping[str, object]) -> str:
    """Content+nonce-addressed evidence identity.

    payload must contain:
      - "source": str
      - "destination": str
      - "sha256": str
      - "size_bytes": int
      - "nonce": str  (caller-supplied, sub-second resolution, e.g. UUID4 hex)
    """
    body = {
        "source": payload["source"],
        "destination": payload["destination"],
        "sha256": payload["sha256"],
        "size_bytes": payload["size_bytes"],
        "nonce": payload["nonce"],
    }
    return f"operator-transfer-evidence-{canonical_mapping_digest(body)}"


@dataclass(frozen=True)
class OperatorTransferEvidence:
    """Evidence for one completed transfer."""

    evidence_id: str
    mode: str
    operator_initiated: bool
    authority_reference: str | None
    authority_digest: str | None
    override_destination_prefix: str | None
    item: OperatorTransferItem
    transfer_tool: str
    transfer_argv: tuple[str, ...]
    transfer_returncode: int
    transfer_elapsed_s: float
    verified_size_bytes: int
    verified_sha256: str
    transferred_at: str
    original_evidence_preserved: bool
    evidence_nonce: str = field(compare=False)
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "OperatorTransferEvidence")
        if self.mode not in _MODES:
            raise ValueError(f"OperatorTransferEvidence unsupported mode: {self.mode!r}")
        if self.operator_initiated is not True:
            raise ValueError("OperatorTransferEvidence operator_initiated must be True")
        if not self.evidence_id:
            raise ValueError("OperatorTransferEvidence evidence_id must be non-empty")
        _validate_non_negative_int(self.verified_size_bytes, "OperatorTransferEvidence verified_size_bytes")
        if _SHA256.fullmatch(self.verified_sha256) is None:
            raise ValueError("OperatorTransferEvidence verified_sha256 must be 64 lowercase hex characters")
        _validate_timestamp(self.transferred_at, "OperatorTransferEvidence transferred_at")
        if not self.evidence_nonce:
            raise ValueError("OperatorTransferEvidence evidence_nonce must be non-empty")
        # The verified content must agree with the declared item content; an
        # evidence record whose verification fields contradict its own item is
        # tampered or corrupt and must never deserialize as valid.
        if self.verified_size_bytes != self.item.size_bytes:
            raise ValueError("OperatorTransferEvidence verified_size_bytes must match item.size_bytes")
        if self.verified_sha256 != self.item.sha256:
            raise ValueError("OperatorTransferEvidence verified_sha256 must match item.sha256")
        expected_id = operator_transfer_evidence_id(self.to_mapping_for_id())
        if self.evidence_id != expected_id:
            raise ValueError("OperatorTransferEvidence evidence_id does not match its content+nonce")

    def to_mapping_for_id(self) -> dict[str, object]:
        return {
            "source": self.item.source,
            "destination": self.item.destination,
            "sha256": self.item.sha256,
            "size_bytes": self.item.size_bytes,
            "nonce": self.evidence_nonce,
        }

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "mode": self.mode,
            "operator_initiated": self.operator_initiated,
            "authority_reference": self.authority_reference,
            "authority_digest": self.authority_digest,
            "override_destination_prefix": self.override_destination_prefix,
            "item": self.item.to_mapping(),
            "transfer_tool": self.transfer_tool,
            "transfer_argv": list(self.transfer_argv),
            "transfer_returncode": self.transfer_returncode,
            "transfer_elapsed_s": self.transfer_elapsed_s,
            "verified_size_bytes": self.verified_size_bytes,
            "verified_sha256": self.verified_sha256,
            "transferred_at": self.transferred_at,
            "original_evidence_preserved": self.original_evidence_preserved,
            "evidence_nonce": self.evidence_nonce,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), sort_keys=True, indent=2)


def operator_transfer_evidence_from_mapping(payload: Mapping[str, object]) -> OperatorTransferEvidence:
    """Deserialize an :class:`OperatorTransferEvidence` from a mapping."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "evidence_id",
            "mode",
            "operator_initiated",
            "authority_reference",
            "authority_digest",
            "override_destination_prefix",
            "item",
            "transfer_tool",
            "transfer_argv",
            "transfer_returncode",
            "transfer_elapsed_s",
            "verified_size_bytes",
            "verified_sha256",
            "transferred_at",
            "original_evidence_preserved",
            "evidence_nonce",
        },
        "OperatorTransferEvidence",
    )
    schema_version = _required_int(payload, "schema_version")
    _validate_direct_schema_version(schema_version, "OperatorTransferEvidence")
    evidence_id = _required_str(payload, "evidence_id")
    mode = _required_str(payload, "mode")
    operator_initiated = payload.get("operator_initiated")
    if not isinstance(operator_initiated, bool):
        raise ValueError("OperatorTransferEvidence operator_initiated must be a bool")
    authority_reference = _optional_str(payload, "authority_reference")
    authority_digest = _optional_str(payload, "authority_digest")
    override_destination_prefix = _optional_str(payload, "override_destination_prefix")
    item = operator_transfer_item_from_mapping(_required_mapping(payload, "item"))
    transfer_tool = _required_str(payload, "transfer_tool")
    transfer_argv_raw = payload.get("transfer_argv")
    if not isinstance(transfer_argv_raw, list | tuple) or not all(isinstance(a, str) for a in transfer_argv_raw):
        raise ValueError("OperatorTransferEvidence transfer_argv must be a list of strings")
    transfer_argv = tuple(transfer_argv_raw)
    transfer_returncode = _required_int(payload, "transfer_returncode")
    transfer_elapsed_s_raw = payload.get("transfer_elapsed_s")
    if not isinstance(transfer_elapsed_s_raw, int | float) or isinstance(transfer_elapsed_s_raw, bool):
        raise ValueError("OperatorTransferEvidence transfer_elapsed_s must be a number")
    transfer_elapsed_s = float(transfer_elapsed_s_raw)
    verified_size_bytes = _required_int(payload, "verified_size_bytes")
    verified_sha256 = _required_str(payload, "verified_sha256")
    transferred_at = _required_str(payload, "transferred_at")
    original_evidence_preserved = payload.get("original_evidence_preserved")
    if not isinstance(original_evidence_preserved, bool):
        raise ValueError("OperatorTransferEvidence original_evidence_preserved must be a bool")
    evidence_nonce = _required_str(payload, "evidence_nonce")
    return OperatorTransferEvidence(
        schema_version=schema_version,
        evidence_id=evidence_id,
        mode=mode,
        operator_initiated=operator_initiated,
        authority_reference=authority_reference,
        authority_digest=authority_digest,
        override_destination_prefix=override_destination_prefix,
        item=item,
        transfer_tool=transfer_tool,
        transfer_argv=transfer_argv,
        transfer_returncode=transfer_returncode,
        transfer_elapsed_s=transfer_elapsed_s,
        verified_size_bytes=verified_size_bytes,
        verified_sha256=verified_sha256,
        transferred_at=transferred_at,
        original_evidence_preserved=original_evidence_preserved,
        evidence_nonce=evidence_nonce,
    )


def operator_transfer_evidence_from_json(text: str) -> OperatorTransferEvidence:
    """Deserialize an :class:`OperatorTransferEvidence` from a JSON string."""
    payload = json.loads(text)
    if not isinstance(payload, Mapping):
        raise ValueError("OperatorTransferEvidence JSON must be a mapping")
    return operator_transfer_evidence_from_mapping(payload)


# ---------------------------------------------------------------------------
# Destination derivation and tool dispatch
# ---------------------------------------------------------------------------


def resolve_transfer_destination(
    *,
    location: VerifiedLocalBundledArtifactLocation | VerifiedRemoteBundledArtifactLocation,
    s3_prefix: str | None,
    override_prefix: str | None,
) -> str:
    """Derive the content-addressed destination object key.

    If ``override_prefix`` is given, it is validated and used. Otherwise
    ``s3_prefix`` is required and validated. Raises ``ValueError``
    on invalid prefix or when both are ``None``.
    """
    if override_prefix is not None:
        normalized = _validate_s3_prefix(override_prefix)
    elif s3_prefix is not None:
        normalized = _validate_s3_prefix(s3_prefix)
    else:
        raise ValueError("resolve_transfer_destination requires either override_prefix or s3_prefix")
    return f"{normalized}/{location.lz4_sha256}.tar.lz4"


def resolve_execution_tool(destination: str) -> str:
    """Derive the transfer tool from the destination URI scheme."""
    if destination.startswith("s3://"):
        return "s5cmd"
    if destination.startswith("gs://"):
        raise ValueError("GCS destinations are not supported in R1; use s3:// destinations")
    raise ValueError(f"unsupported destination scheme for operator transfer: {destination}")


# ---------------------------------------------------------------------------
# Plan builders
# ---------------------------------------------------------------------------


def build_plan_referenced_transfer_plan(
    *,
    phase_plan_path: Path,
    s3_prefix: str | None = None,
    override_prefix: str | None = None,
    dry_run: bool = True,
) -> OperatorTransferPlan:
    """Build a transfer plan from a FoldingPhasePlan authority (read-only)."""
    data = yaml.safe_load(phase_plan_path.read_bytes())
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {phase_plan_path}")
    plan = folding_phase_plan_from_mapping(data)
    decision = resolve_seam_transfer_decision(policy=plan.payload.transport)
    location = plan.input_location
    destination = resolve_transfer_destination(
        location=location,
        s3_prefix=s3_prefix,
        override_prefix=override_prefix,
    )
    if isinstance(location, VerifiedLocalBundledArtifactLocation):
        source = location.bundle_path
        source_kind = "verified-local-bundled"
        description = f"Local MSA-set bundle from {phase_plan_path.name}"
    else:
        source = location.bundle_uri
        source_kind = "verified-remote-bundled"
        description = f"Remote MSA-set bundle from {phase_plan_path.name}"
    item = OperatorTransferItem(
        source=source,
        destination=destination,
        size_bytes=location.lz4_size_bytes,
        sha256=location.lz4_sha256,
        artifact_set_id=location.artifact_set_id,
        artifact_location_id=location.artifact_location_id,
        source_kind=source_kind,
        description=description,
    )
    return OperatorTransferPlan(
        mode="plan-referenced",
        items=(item,),
        decision=decision,
        operator_initiated=True,
        authority_reference=str(phase_plan_path),
        authority_digest=plan.digest,
        override_destination_prefix=override_prefix,
        dry_run=dry_run,
    )


def build_manual_transfer_plan(
    *,
    source: str,
    destination: str,
    size_bytes: int,
    sha256: str,
    s3_prefix: str | None = None,
    override_prefix: str | None = None,
    dry_run: bool = True,
) -> OperatorTransferPlan:
    """Build a transfer plan from explicit paths."""
    if override_prefix is not None:
        raise ValueError("build_manual_transfer_plan does not accept override_prefix")
    decision = resolve_seam_transfer_decision(policy="publish-to-s3")
    item = OperatorTransferItem(
        source=source,
        destination=destination,
        size_bytes=size_bytes,
        sha256=sha256,
        artifact_set_id=None,
        artifact_location_id=None,
        source_kind="explicit-path",
        description="Manual operator transfer",
    )
    return OperatorTransferPlan(
        mode="manual",
        items=(item,),
        decision=decision,
        operator_initiated=True,
        authority_reference=None,
        authority_digest=None,
        override_destination_prefix=None,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Shared mapping helpers
# ---------------------------------------------------------------------------


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {record_name} field(s): {', '.join(unknown)}")


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


__all__ = [
    "OperatorTransferEvidence",
    "OperatorTransferItem",
    "OperatorTransferPlan",
    "SeamTransferDecision",
    "build_manual_transfer_plan",
    "build_plan_referenced_transfer_plan",
    "operator_transfer_evidence_from_json",
    "operator_transfer_evidence_from_mapping",
    "operator_transfer_evidence_id",
    "operator_transfer_item_from_mapping",
    "operator_transfer_plan_from_json",
    "operator_transfer_plan_from_mapping",
    "resolve_execution_tool",
    "resolve_seam_transfer_decision",
    "resolve_transfer_destination",
    "seam_transfer_decision_from_mapping",
]
