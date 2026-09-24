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

"""Shared provisioning contract records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from bspp.orchestration.contract.source_package import SourcePackageIdentity
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version


@dataclass(frozen=True)
class SourceBundlePlan:
    """Planned Source Bundle identity and target location."""

    source_repo: str
    source_state: str
    commit: str
    tree: str
    dirty_evidence_hash: str | None
    bundle_id: str
    target_root: str
    target_path: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML-ready data."""
        return {
            "schema_version": self.schema_version,
            "source_repo": self.source_repo,
            "source_state": self.source_state,
            "commit": self.commit,
            "tree": self.tree,
            "dirty_evidence_hash": self.dirty_evidence_hash,
            "bundle_id": self.bundle_id,
            "target_root": self.target_root,
            "target_path": self.target_path,
        }


@dataclass(frozen=True)
class SourceBundleManifestEntry:
    """One file entry in a Source Bundle manifest."""

    path: str
    size_bytes: int
    sha256: str
    mode: str

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML/JSON-ready data."""
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "mode": self.mode,
        }

    def with_sha256(self, sha256: str) -> SourceBundleManifestEntry:
        """Return a copy with a replaced checksum for testable validation paths."""
        return replace(self, sha256=sha256)


@dataclass(frozen=True)
class SourceBundleBuildRecord:
    """Local Source Bundle archive build evidence."""

    source_bundle: SourceBundlePlan
    archive_path: Path
    archive_size_bytes: int
    archive_sha256: str
    manifest_entries: tuple[SourceBundleManifestEntry, ...]
    manifest_sha256: str
    dirty_source_policy: str
    evidence_path: Path
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML/JSON-ready data."""
        return {
            "source_bundle_build": {
                "schema_version": self.schema_version,
                "source_bundle": self.source_bundle.to_mapping(),
                "archive_path": str(self.archive_path),
                "archive_size_bytes": self.archive_size_bytes,
                "archive_sha256": self.archive_sha256,
                "manifest_sha256": self.manifest_sha256,
                "dirty_source_policy": self.dirty_source_policy,
                "manifest_entries": [entry.to_mapping() for entry in self.manifest_entries],
                "evidence_path": str(self.evidence_path),
            }
        }

    def with_manifest_entries(
        self,
        manifest_entries: tuple[SourceBundleManifestEntry, ...],
    ) -> SourceBundleBuildRecord:
        """Return a copy with manifest entries and digest replaced together."""
        return replace(
            self,
            manifest_entries=manifest_entries,
            manifest_sha256=source_bundle_manifest_digest(manifest_entries),
        )


@dataclass(frozen=True)
class SourceBundleArchiveVerification:
    """Verification result for a local Source Bundle archive."""

    ok: bool
    manifest_sha256: str
    issues: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML/JSON-ready data."""
        return {
            "source_bundle_archive_verification": {
                "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
                "ok": self.ok,
                "manifest_sha256": self.manifest_sha256,
                "issues": list(self.issues),
            }
        }


@dataclass(frozen=True)
class SourceBundleStageRecord:
    """Cluster staging and verification evidence for a Source Bundle."""

    source_bundle: SourceBundlePlan
    archive_path: Path
    target_path: str
    archive_sha256: str
    verified_sha256: str
    commands: tuple[tuple[str, ...], ...]
    evidence_path: Path
    staged: bool = True
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML/JSON-ready data."""
        return {
            "source_bundle_stage": {
                "schema_version": self.schema_version,
                "source_bundle": self.source_bundle.to_mapping(),
                "archive_path": str(self.archive_path),
                "target_path": self.target_path,
                "archive_sha256": self.archive_sha256,
                "verified_sha256": self.verified_sha256,
                "staged": self.staged,
                "commands": [list(command) for command in self.commands],
                "evidence_path": str(self.evidence_path),
            }
        }


def source_bundle_manifest_digest(entries: tuple[SourceBundleManifestEntry, ...]) -> str:
    """Return a stable digest over Source Bundle manifest entries."""
    payload = json.dumps([entry.to_mapping() for entry in entries], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def source_package_identity_from_source_bundle(record: SourceBundleBuildRecord) -> SourcePackageIdentity:
    """Bind a legacy Source Bundle build record to the governed package identity model."""
    return SourcePackageIdentity(
        format="safe-tar-v1",
        verifier="safe-tar-v1",
        package_path=record.archive_path.absolute(),
        package_size_bytes=record.archive_size_bytes,
        package_sha256=record.archive_sha256,
        manifest_sha256=record.manifest_sha256,
        commit=record.source_bundle.commit,
        tree=record.source_bundle.tree,
    )


@dataclass(frozen=True)
class RuntimeImagePlan:
    """Planned Execution Runtime Image cache state."""

    reference: str
    cache_root: str
    cache_path: str
    cache_status: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML-ready data."""
        return {
            "schema_version": self.schema_version,
            "reference": self.reference,
            "cache_root": self.cache_root,
            "cache_path": self.cache_path,
            "cache_status": self.cache_status,
        }


@dataclass(frozen=True)
class RuntimeImageCacheCheckRecord:
    """Runtime Image cache probe evidence."""

    runtime_image: RuntimeImagePlan
    status: str
    checked: bool
    probe_command: tuple[str, ...]
    probe_returncode: int
    evidence_path: Path
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML/JSON-ready data."""
        return {
            "runtime_image_cache_check": {
                "schema_version": self.schema_version,
                "runtime_image": self.runtime_image.to_mapping(),
                "status": self.status,
                "checked": self.checked,
                "probe_command": list(self.probe_command),
                "probe_returncode": self.probe_returncode,
                "evidence_path": str(self.evidence_path),
            }
        }


@dataclass(frozen=True)
class RuntimeImageProvisionRecord:
    """Runtime Image cache provisioning evidence."""

    runtime_image: RuntimeImagePlan
    cache_check: RuntimeImageCacheCheckRecord
    status: str
    evidence_path: Path
    script_path: Path | None = None
    submit_command: tuple[str, ...] = ()
    job_id: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML/JSON-ready data."""
        return {
            "runtime_image_provisioning": {
                "schema_version": self.schema_version,
                "runtime_image": self.runtime_image.to_mapping(),
                "cache_check": self.cache_check.to_mapping()["runtime_image_cache_check"],
                "status": self.status,
                "script_path": str(self.script_path) if self.script_path is not None else None,
                "submit_command": list(self.submit_command),
                "job_id": self.job_id,
                "evidence_path": str(self.evidence_path),
            }
        }


@dataclass(frozen=True)
class ProvisioningPlan:
    """Provisioning-only plan for one Run Plan."""

    cluster: str
    run_kind: str
    source_bundle: SourceBundlePlan
    runtime_image: RuntimeImagePlan
    payload_bytes_moved: bool
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, object]:
        """Return deterministic YAML-ready data."""
        return {
            "provisioning_plan": {
                "schema_version": self.schema_version,
                "cluster": self.cluster,
                "run_kind": self.run_kind,
                "source_bundle": self.source_bundle.to_mapping(),
                "runtime_image": self.runtime_image.to_mapping(),
                "payload_bytes_moved": self.payload_bytes_moved,
            }
        }


def source_bundle_plan_from_mapping(payload: Mapping[str, object]) -> SourceBundlePlan:
    """Parse a Source Bundle plan from a YAML/JSON mapping."""
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="SourceBundle")
    return SourceBundlePlan(
        schema_version=schema_version,
        source_repo=_required_str(payload, "source_repo"),
        source_state=_required_str(payload, "source_state"),
        commit=_required_str(payload, "commit"),
        tree=_required_str(payload, "tree"),
        dirty_evidence_hash=_optional_str(payload, "dirty_evidence_hash"),
        bundle_id=_required_str(payload, "bundle_id"),
        target_root=_required_str(payload, "target_root"),
        target_path=_required_str(payload, "target_path"),
    )


def runtime_image_plan_from_mapping(payload: Mapping[str, object]) -> RuntimeImagePlan:
    """Parse a runtime image plan from a YAML/JSON mapping."""
    schema_version = validate_schema_version(payload.get("schema_version"), record_name="RuntimeImage")
    return RuntimeImagePlan(
        schema_version=schema_version,
        reference=_required_str(payload, "reference"),
        cache_root=_required_str(payload, "cache_root"),
        cache_path=_required_str(payload, "cache_path"),
        cache_status=_required_str(payload, "cache_status"),
    )


def provisioning_plan_from_mapping(payload: Mapping[str, object]) -> ProvisioningPlan:
    """Parse a provisioning plan from an outer or inner YAML/JSON mapping."""
    inner = payload.get("provisioning_plan", payload)
    if not isinstance(inner, Mapping):
        msg = "provisioning_plan must be a mapping"
        raise ValueError(msg)
    schema_version = validate_schema_version(inner.get("schema_version"), record_name="ProvisioningPlan")
    source_bundle = _required_mapping(inner, "source_bundle")
    runtime_image = _required_mapping(inner, "runtime_image")
    return ProvisioningPlan(
        schema_version=schema_version,
        cluster=_required_str(inner, "cluster"),
        run_kind=_required_str(inner, "run_kind"),
        source_bundle=source_bundle_plan_from_mapping(source_bundle),
        runtime_image=runtime_image_plan_from_mapping(runtime_image),
        payload_bytes_moved=_required_bool(inner, "payload_bytes_moved"),
    )


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be a mapping"
        raise ValueError(msg)
    return value


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None or isinstance(value, str):
        return value
    msg = f"{key} must be a string or null"
    raise ValueError(msg)


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean"
        raise ValueError(msg)
    return value


__all__ = [
    "ProvisioningPlan",
    "RuntimeImageCacheCheckRecord",
    "RuntimeImagePlan",
    "RuntimeImageProvisionRecord",
    "SourceBundleArchiveVerification",
    "SourceBundleBuildRecord",
    "SourceBundleManifestEntry",
    "SourceBundlePlan",
    "SourceBundleStageRecord",
    "provisioning_plan_from_mapping",
    "runtime_image_plan_from_mapping",
    "source_bundle_manifest_digest",
    "source_bundle_plan_from_mapping",
    "source_package_identity_from_source_bundle",
]
