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

"""Strict identities for the preprocessing Execution Runtime image."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

from bspp.orchestration.contract.preprocessing_action import PREPROCESSING_COMMAND_ORDER
from bspp.orchestration.contract.preprocessing_identity import PREPROCESSING_ADAPTER_VERSION
from bspp.orchestration.contract.slurm import validate_exact_slurm_node
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION, validate_schema_version

PREPROCESSING_RUNTIME_COMMAND = ("bspp-orchestration-runtime", "preprocessing", "execute-chunk")
LEGACY_PREPROCESSING_RUNTIME_COMMAND = ("afcdb-orchestration-runtime", "preprocessing", "execute-chunk")

_RUNTIME_CONTRACT_PAYLOAD: dict[str, object] = {
    "adapter_version": PREPROCESSING_ADAPTER_VERSION,
    "command_order": list(PREPROCESSING_COMMAND_ORDER),
    "evidence_record": "PreprocessingChunkActionEvidence",
    "raw_search_closure": "isolate-validate-publish-paired-m-plus-m",
    # The contract identity is a global stable identifier bound to the pre-rename
    # runtime command: historical evidence depends on it byte-identically, while
    # the rendered action command uses PREPROCESSING_RUNTIME_COMMAND (renamed).
    "runtime_command": list(LEGACY_PREPROCESSING_RUNTIME_COMMAND),
    "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
}
PREPROCESSING_RUNTIME_CONTRACT_ID = hashlib.sha256(
    json.dumps(_RUNTIME_CONTRACT_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

QualificationStatus = Literal["submitted", "qualified"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_OCI_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_NORMALIZED_TOOL_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
_RSYNC_VERSION_BANNER = re.compile(
    r"rsync\s+version\s+(?P<version>[0-9]+\.[0-9]+\.[0-9]+)\s+protocol\s+version\s+[0-9]+"
)


def normalize_rsync_version(value: str) -> str:
    """Return one normalized rsync version from a manifest value or exact banner line."""
    if _NORMALIZED_TOOL_VERSION.fullmatch(value):
        return value
    match = _RSYNC_VERSION_BANNER.fullmatch(value)
    if match is None:
        raise ValueError("rsync_version must be normalized x.y.z or an exact rsync first banner line")
    return match.group("version")


@dataclass(frozen=True)
class PreprocessingRuntimeToolEvidence:
    python_version: str
    contract_version: str
    runtime_version: str
    control_version: str | None
    mmseqs_version: str
    colabfold_version: str
    rsync_version: str
    tar_version: str
    lz4_version: str
    flock_version: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeToolEvidence")
        for field_name, value in self.to_mapping().items():
            if field_name == "schema_version":
                continue
            if field_name == "control_version" and value is None:
                continue
            _validate_nonempty(cast("str", value), field_name)
        if normalize_rsync_version(self.rsync_version) != self.rsync_version:
            raise ValueError("preprocessing tool evidence rsync_version must be normalized")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "python_version": self.python_version,
            "contract_version": self.contract_version,
            "runtime_version": self.runtime_version,
            "control_version": self.control_version,
            "mmseqs_version": self.mmseqs_version,
            "colabfold_version": self.colabfold_version,
            "rsync_version": self.rsync_version,
            "tar_version": self.tar_version,
            "lz4_version": self.lz4_version,
            "flock_version": self.flock_version,
        }


@dataclass(frozen=True)
class PreprocessingRuntimeImageEvidence:
    manifest_path: str
    manifest_sha256: str
    cluster_image_sha256: str
    oci_digest: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeImageEvidence")
        _validate_nonempty(self.manifest_path, "manifest_path")
        _validate_sha256(self.manifest_sha256, "manifest_sha256")
        _validate_sha256(self.cluster_image_sha256, "cluster_image_sha256")
        if not _OCI_DIGEST.fullmatch(self.oci_digest):
            raise ValueError("oci_digest must be sha256:<64 lowercase hex>")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "cluster_image_sha256": self.cluster_image_sha256,
            "oci_digest": self.oci_digest,
        }


@dataclass(frozen=True)
class PreprocessingRuntimeSourceEvidence:
    bundle_id: str
    bundle_path: str
    bundle_sha256: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeSourceEvidence")
        _validate_nonempty(self.bundle_id, "bundle_id")
        _validate_nonempty(self.bundle_path, "bundle_path")
        _validate_sha256(self.bundle_sha256, "bundle_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "bundle_path": self.bundle_path,
            "bundle_sha256": self.bundle_sha256,
        }


@dataclass(frozen=True)
class PreprocessingRuntimeGpuEvidence:
    """GPU name/driver output captured by the allocated scheduled host before container entry."""

    nvidia_smi: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeGpuEvidence")
        _validate_nonempty(self.nvidia_smi, "nvidia_smi")

    def to_mapping(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, "nvidia_smi": self.nvidia_smi}


@dataclass(frozen=True)
class PreprocessingRuntimeSmokeEvidence:
    """Strict observations from the scheduled image smoke and its allocated-host GPU probe.

    GPU evidence records host-visible name/driver output relayed into the sealed
    image. It does not assert image-native device, driver-library, or scientific
    CUDA-kernel availability.
    """

    runtime_command: tuple[str, ...]
    runtime_contract_id: str
    adapter_version: str
    command_order: tuple[str, ...]
    action_evidence_sha256: str
    tools: PreprocessingRuntimeToolEvidence
    image: PreprocessingRuntimeImageEvidence
    source: PreprocessingRuntimeSourceEvidence
    gpu: PreprocessingRuntimeGpuEvidence
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeSmokeEvidence")
        if self.runtime_command not in (PREPROCESSING_RUNTIME_COMMAND, LEGACY_PREPROCESSING_RUNTIME_COMMAND):
            raise ValueError("preprocessing smoke runtime command does not match")
        if self.runtime_contract_id != PREPROCESSING_RUNTIME_CONTRACT_ID:
            raise ValueError("preprocessing smoke runtime contract does not match")
        if self.adapter_version != PREPROCESSING_ADAPTER_VERSION:
            raise ValueError("preprocessing smoke adapter version does not match")
        if self.command_order != PREPROCESSING_COMMAND_ORDER:
            raise ValueError("preprocessing smoke command order does not match")
        _validate_sha256(self.action_evidence_sha256, "action_evidence_sha256")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_command": list(self.runtime_command),
            "runtime_contract_id": self.runtime_contract_id,
            "adapter_version": self.adapter_version,
            "command_order": list(self.command_order),
            "action_evidence_sha256": self.action_evidence_sha256,
            "tools": self.tools.to_mapping(),
            "image": self.image.to_mapping(),
            "source": self.source.to_mapping(),
            "gpu": self.gpu.to_mapping(),
        }


@dataclass(frozen=True)
class PreprocessingRuntimeImageIdentity:
    """Immutable source and toolchain identity baked into one image."""

    source_commit: str
    image_lock_sha256: str
    contract_wheel_sha256: str
    runtime_wheel_sha256: str
    colabfold_version: str
    mmseqs_version: str
    rsync_version: str
    cuda_version: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION
    control_wheel_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeImageIdentity")
        if not _GIT_COMMIT.fullmatch(self.source_commit):
            raise ValueError("preprocessing runtime source_commit must be a full lowercase Git SHA-1")
        for field_name, value in (
            ("image_lock_sha256", self.image_lock_sha256),
            ("contract_wheel_sha256", self.contract_wheel_sha256),
            ("runtime_wheel_sha256", self.runtime_wheel_sha256),
        ):
            _validate_sha256(value, field_name)
        if self.control_wheel_sha256 is not None:
            _validate_sha256(self.control_wheel_sha256, "control_wheel_sha256")
        for field_name, value in (
            ("colabfold_version", self.colabfold_version),
            ("mmseqs_version", self.mmseqs_version),
            ("rsync_version", self.rsync_version),
            ("cuda_version", self.cuda_version),
        ):
            _validate_nonempty(value, field_name)
        if normalize_rsync_version(self.rsync_version) != self.rsync_version:
            raise ValueError("preprocessing image rsync_version must be normalized")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "image_lock_sha256": self.image_lock_sha256,
            "contract_wheel_sha256": self.contract_wheel_sha256,
            "runtime_wheel_sha256": self.runtime_wheel_sha256,
            "colabfold_version": self.colabfold_version,
            "mmseqs_version": self.mmseqs_version,
            "rsync_version": self.rsync_version,
            "cuda_version": self.cuda_version,
        }
        if self.control_wheel_sha256 is not None:
            result["control_wheel_sha256"] = self.control_wheel_sha256
        return result


@dataclass(frozen=True)
class PreprocessingRuntimeQualificationTuple:
    """Exact reusable preprocessing runtime setup that was qualified."""

    cluster_profile: str
    scheduling_class: str
    gpu_worker_gres: str
    cluster_image_path: str
    cluster_image_sha256: str
    oci_digest: str
    source_bundle_id: str
    source_bundle_path: str
    source_bundle_sha256: str
    runtime_contract_id: str
    adapter_version: str
    image_identity: PreprocessingRuntimeImageIdentity
    nodelist: str | None = None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeQualificationTuple")
        for field_name, value in (
            ("cluster_profile", self.cluster_profile),
            ("gpu_worker_gres", self.gpu_worker_gres),
            ("cluster_image_path", self.cluster_image_path),
            ("source_bundle_id", self.source_bundle_id),
            ("source_bundle_path", self.source_bundle_path),
        ):
            _validate_nonempty(value, field_name)
        if self.scheduling_class != "gpu_worker":
            raise ValueError("preprocessing Runtime Qualification scheduling_class must be 'gpu_worker'")
        _validate_sha256(self.cluster_image_sha256, "cluster_image_sha256")
        _validate_sha256(self.source_bundle_sha256, "source_bundle_sha256")
        if not _OCI_DIGEST.fullmatch(self.oci_digest):
            raise ValueError("preprocessing runtime oci_digest must be sha256:<64 lowercase hex>")
        if self.runtime_contract_id != PREPROCESSING_RUNTIME_CONTRACT_ID:
            raise ValueError("preprocessing runtime contract identity is unsupported")
        if self.adapter_version != PREPROCESSING_ADAPTER_VERSION:
            raise ValueError("preprocessing adapter version is unsupported")
        validate_exact_slurm_node(self.nodelist, field_name="preprocessing qualification nodelist")

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "cluster_profile": self.cluster_profile,
            "scheduling_class": self.scheduling_class,
            "gpu_worker_gres": self.gpu_worker_gres,
            "cluster_image_path": self.cluster_image_path,
            "cluster_image_sha256": self.cluster_image_sha256,
            "oci_digest": self.oci_digest,
            "source_bundle_id": self.source_bundle_id,
            "source_bundle_path": self.source_bundle_path,
            "source_bundle_sha256": self.source_bundle_sha256,
            "runtime_contract_id": self.runtime_contract_id,
            "adapter_version": self.adapter_version,
            "image_identity": self.image_identity.to_mapping(),
        }
        if self.nodelist is not None:
            result["nodelist"] = self.nodelist
        return result


def validate_preprocessing_runtime_tool_evidence(
    image_identity: PreprocessingRuntimeImageIdentity,
    observed: PreprocessingRuntimeToolEvidence,
) -> None:
    """Bind smoke-observed tool versions to the baked image identity."""
    if (
        image_identity.mmseqs_version not in observed.mmseqs_version
        or observed.colabfold_version != image_identity.colabfold_version
        or observed.rsync_version != image_identity.rsync_version
    ):
        raise ValueError("qualified preprocessing observed tool identity does not match its baked image identity")
    if image_identity.control_wheel_sha256 is not None and observed.control_version is None:
        raise ValueError(
            "qualified preprocessing tool evidence must attest a Control version when the image bakes Control"
        )


@dataclass(frozen=True)
class PreprocessingRuntimeQualificationRecord:
    """Submitted or smoke-written evidence for one exact tuple."""

    status: QualificationStatus
    tuple_id: str
    qualification_tuple: PreprocessingRuntimeQualificationTuple
    submitted_at: str
    qualified_at: str | None
    expires_at: str | None
    job_id: str | None
    smoke_evidence: PreprocessingRuntimeSmokeEvidence | None
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "PreprocessingRuntimeQualificationRecord")
        if self.status not in {"submitted", "qualified"}:
            raise ValueError(f"unsupported preprocessing qualification status: {self.status!r}")
        if self.tuple_id != preprocessing_runtime_tuple_id(self.qualification_tuple):
            raise ValueError("preprocessing qualification tuple_id does not match its tuple")
        submitted_at = _parse_timestamp(self.submitted_at, "submitted_at")
        if self.status == "submitted":
            if self.qualified_at is not None or self.expires_at is not None or self.smoke_evidence is not None:
                raise ValueError("submitted preprocessing qualification cannot contain qualified evidence")
            if self.job_id == "":
                raise ValueError("submitted preprocessing qualification job_id must be non-empty when present")
            return
        if self.job_id is None or self.qualified_at is None or self.expires_at is None or self.smoke_evidence is None:
            raise ValueError("qualified preprocessing qualification requires job, timestamps, and smoke evidence")
        _validate_nonempty(self.job_id, "job_id")
        qualified_at = _parse_timestamp(self.qualified_at, "qualified_at")
        expires_at = _parse_timestamp(self.expires_at, "expires_at")
        if qualified_at < submitted_at:
            raise ValueError("preprocessing qualification qualified_at must not precede submitted_at")
        if expires_at <= qualified_at:
            raise ValueError("preprocessing qualification expires_at must be after qualified_at")
        if (
            self.smoke_evidence.runtime_contract_id != self.qualification_tuple.runtime_contract_id
            or self.smoke_evidence.adapter_version != self.qualification_tuple.adapter_version
            or self.smoke_evidence.image.cluster_image_sha256 != self.qualification_tuple.cluster_image_sha256
            or self.smoke_evidence.image.oci_digest != self.qualification_tuple.oci_digest
            or self.smoke_evidence.source.bundle_id != self.qualification_tuple.source_bundle_id
            or self.smoke_evidence.source.bundle_path != self.qualification_tuple.source_bundle_path
            or self.smoke_evidence.source.bundle_sha256 != self.qualification_tuple.source_bundle_sha256
        ):
            raise ValueError("qualified preprocessing smoke evidence does not match its tuple")
        validate_preprocessing_runtime_tool_evidence(
            self.qualification_tuple.image_identity,
            self.smoke_evidence.tools,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "tuple_id": self.tuple_id,
            "qualification_tuple": self.qualification_tuple.to_mapping(),
            "submitted_at": self.submitted_at,
            "qualified_at": self.qualified_at,
            "expires_at": self.expires_at,
            "job_id": self.job_id,
            "smoke_evidence": None if self.smoke_evidence is None else self.smoke_evidence.to_mapping(),
        }


@dataclass(frozen=True)
class QualifiedPreprocessingRuntimeSelection:
    """Immutable qualified runtime selection embedded in a Phase RunSpec."""

    qualification_tuple: PreprocessingRuntimeQualificationTuple
    qualification_record_path: str
    qualified_at: str
    expires_at: str
    schema_version: int = CURRENT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_direct_schema_version(self.schema_version, "QualifiedPreprocessingRuntimeSelection")
        _validate_nonempty(self.qualification_record_path, "qualification_record_path")
        qualified_at = _parse_timestamp(self.qualified_at, "qualified_at")
        expires_at = _parse_timestamp(self.expires_at, "expires_at")
        if expires_at <= qualified_at:
            raise ValueError("qualified preprocessing selection must expire after qualification")

    @property
    def tuple_id(self) -> str:
        return preprocessing_runtime_tuple_id(self.qualification_tuple)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "qualification_tuple": self.qualification_tuple.to_mapping(),
            "qualification_record_path": self.qualification_record_path,
            "qualified_at": self.qualified_at,
            "expires_at": self.expires_at,
        }


def preprocessing_runtime_tuple_id(value: PreprocessingRuntimeQualificationTuple) -> str:
    """Return the canonical content identity for one qualification tuple."""
    encoded = json.dumps(value.to_mapping(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preprocessing_runtime_image_identity_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRuntimeImageIdentity:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "source_commit",
            "image_lock_sha256",
            "contract_wheel_sha256",
            "runtime_wheel_sha256",
            "control_wheel_sha256",
            "colabfold_version",
            "mmseqs_version",
            "rsync_version",
            "cuda_version",
        },
        "PreprocessingRuntimeImageIdentity",
    )
    return PreprocessingRuntimeImageIdentity(
        schema_version=_required_schema_version(payload, "PreprocessingRuntimeImageIdentity"),
        source_commit=_required_str(payload, "source_commit"),
        image_lock_sha256=_required_str(payload, "image_lock_sha256"),
        contract_wheel_sha256=_required_str(payload, "contract_wheel_sha256"),
        runtime_wheel_sha256=_required_str(payload, "runtime_wheel_sha256"),
        control_wheel_sha256=_optional_str(payload, "control_wheel_sha256"),
        colabfold_version=_required_str(payload, "colabfold_version"),
        mmseqs_version=_required_str(payload, "mmseqs_version"),
        rsync_version=_required_str(payload, "rsync_version"),
        cuda_version=_required_str(payload, "cuda_version"),
    )


def preprocessing_runtime_qualification_tuple_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRuntimeQualificationTuple:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "cluster_profile",
            "scheduling_class",
            "gpu_worker_gres",
            "cluster_image_path",
            "cluster_image_sha256",
            "oci_digest",
            "source_bundle_id",
            "source_bundle_path",
            "source_bundle_sha256",
            "runtime_contract_id",
            "adapter_version",
            "image_identity",
            "nodelist",
        },
        "PreprocessingRuntimeQualificationTuple",
    )
    return PreprocessingRuntimeQualificationTuple(
        schema_version=_required_schema_version(payload, "PreprocessingRuntimeQualificationTuple"),
        cluster_profile=_required_str(payload, "cluster_profile"),
        scheduling_class=_required_str(payload, "scheduling_class"),
        gpu_worker_gres=_required_str(payload, "gpu_worker_gres"),
        cluster_image_path=_required_str(payload, "cluster_image_path"),
        cluster_image_sha256=_required_str(payload, "cluster_image_sha256"),
        oci_digest=_required_str(payload, "oci_digest"),
        source_bundle_id=_required_str(payload, "source_bundle_id"),
        source_bundle_path=_required_str(payload, "source_bundle_path"),
        source_bundle_sha256=_required_str(payload, "source_bundle_sha256"),
        runtime_contract_id=_required_str(payload, "runtime_contract_id"),
        adapter_version=_required_str(payload, "adapter_version"),
        image_identity=preprocessing_runtime_image_identity_from_mapping(_required_mapping(payload, "image_identity")),
        nodelist=_optional_str(payload, "nodelist"),
    )


def preprocessing_runtime_qualification_record_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRuntimeQualificationRecord:
    """Strict-load one submitted or qualified preprocessing record."""
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "status",
            "tuple_id",
            "qualification_tuple",
            "submitted_at",
            "qualified_at",
            "expires_at",
            "job_id",
            "smoke_evidence",
        },
        "PreprocessingRuntimeQualificationRecord",
    )
    status = _required_str(payload, "status")
    return PreprocessingRuntimeQualificationRecord(
        schema_version=_required_schema_version(payload, "PreprocessingRuntimeQualificationRecord"),
        status=cast("QualificationStatus", status),
        tuple_id=_required_str(payload, "tuple_id"),
        qualification_tuple=preprocessing_runtime_qualification_tuple_from_mapping(
            _required_mapping(payload, "qualification_tuple")
        ),
        submitted_at=_required_str(payload, "submitted_at"),
        qualified_at=_required_optional_str(payload, "qualified_at"),
        expires_at=_required_optional_str(payload, "expires_at"),
        job_id=_required_optional_str(payload, "job_id"),
        smoke_evidence=_optional_smoke_evidence(payload),
    )


def qualified_preprocessing_runtime_selection_from_mapping(
    payload: Mapping[str, object],
) -> QualifiedPreprocessingRuntimeSelection:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "qualification_tuple",
            "qualification_record_path",
            "qualified_at",
            "expires_at",
        },
        "QualifiedPreprocessingRuntimeSelection",
    )
    return QualifiedPreprocessingRuntimeSelection(
        schema_version=_required_schema_version(payload, "QualifiedPreprocessingRuntimeSelection"),
        qualification_tuple=preprocessing_runtime_qualification_tuple_from_mapping(
            _required_mapping(payload, "qualification_tuple")
        ),
        qualification_record_path=_required_str(payload, "qualification_record_path"),
        qualified_at=_required_str(payload, "qualified_at"),
        expires_at=_required_str(payload, "expires_at"),
    )


def preprocessing_runtime_smoke_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRuntimeSmokeEvidence:
    _reject_unknown_fields(
        payload,
        {
            "schema_version",
            "runtime_command",
            "runtime_contract_id",
            "adapter_version",
            "command_order",
            "action_evidence_sha256",
            "tools",
            "image",
            "source",
            "gpu",
        },
        "PreprocessingRuntimeSmokeEvidence",
    )
    return PreprocessingRuntimeSmokeEvidence(
        schema_version=_required_schema_version(payload, "PreprocessingRuntimeSmokeEvidence"),
        runtime_command=_required_str_tuple(payload, "runtime_command"),
        runtime_contract_id=_required_str(payload, "runtime_contract_id"),
        adapter_version=_required_str(payload, "adapter_version"),
        command_order=_required_str_tuple(payload, "command_order"),
        action_evidence_sha256=_required_str(payload, "action_evidence_sha256"),
        tools=preprocessing_runtime_tool_evidence_from_mapping(_required_mapping(payload, "tools")),
        image=preprocessing_runtime_image_evidence_from_mapping(_required_mapping(payload, "image")),
        source=preprocessing_runtime_source_evidence_from_mapping(_required_mapping(payload, "source")),
        gpu=preprocessing_runtime_gpu_evidence_from_mapping(_required_mapping(payload, "gpu")),
    )


def preprocessing_runtime_tool_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRuntimeToolEvidence:
    fields = {
        "schema_version",
        "python_version",
        "contract_version",
        "runtime_version",
        "control_version",
        "control_absent",  # legacy alias: Control was absent, so no version is attested
        "mmseqs_version",
        "colabfold_version",
        "rsync_version",
        "tar_version",
        "lz4_version",
        "flock_version",
    }
    _reject_unknown_fields(payload, fields, "PreprocessingRuntimeToolEvidence")
    if "control_absent" in payload:
        # Legacy alias: Control was absent, so no version is attested. Reject
        # ambiguity (both keys) and invalid legacy values.
        if "control_version" in payload:
            raise ValueError("PreprocessingRuntimeToolEvidence must not carry both control_version and control_absent")
        if payload["control_absent"] != "true":
            raise ValueError('legacy control_absent must be "true" (Control was absent)')
        control_version: str | None = None
    else:
        # Current-schema evidence must attest a non-empty Control version.
        control_version = _required_str(payload, "control_version")
    return PreprocessingRuntimeToolEvidence(
        schema_version=_required_schema_version(payload, "PreprocessingRuntimeToolEvidence"),
        python_version=_required_str(payload, "python_version"),
        contract_version=_required_str(payload, "contract_version"),
        runtime_version=_required_str(payload, "runtime_version"),
        control_version=control_version,
        mmseqs_version=_required_str(payload, "mmseqs_version"),
        colabfold_version=_required_str(payload, "colabfold_version"),
        rsync_version=_required_str(payload, "rsync_version"),
        tar_version=_required_str(payload, "tar_version"),
        lz4_version=_required_str(payload, "lz4_version"),
        flock_version=_required_str(payload, "flock_version"),
    )


def preprocessing_runtime_image_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRuntimeImageEvidence:
    _reject_unknown_fields(
        payload,
        {"schema_version", "manifest_path", "manifest_sha256", "cluster_image_sha256", "oci_digest"},
        "PreprocessingRuntimeImageEvidence",
    )
    return PreprocessingRuntimeImageEvidence(
        schema_version=_required_schema_version(payload, "PreprocessingRuntimeImageEvidence"),
        manifest_path=_required_str(payload, "manifest_path"),
        manifest_sha256=_required_str(payload, "manifest_sha256"),
        cluster_image_sha256=_required_str(payload, "cluster_image_sha256"),
        oci_digest=_required_str(payload, "oci_digest"),
    )


def preprocessing_runtime_source_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRuntimeSourceEvidence:
    _reject_unknown_fields(
        payload,
        {"schema_version", "bundle_id", "bundle_path", "bundle_sha256"},
        "PreprocessingRuntimeSourceEvidence",
    )
    return PreprocessingRuntimeSourceEvidence(
        schema_version=_required_schema_version(payload, "PreprocessingRuntimeSourceEvidence"),
        bundle_id=_required_str(payload, "bundle_id"),
        bundle_path=_required_str(payload, "bundle_path"),
        bundle_sha256=_required_str(payload, "bundle_sha256"),
    )


def preprocessing_runtime_gpu_evidence_from_mapping(
    payload: Mapping[str, object],
) -> PreprocessingRuntimeGpuEvidence:
    _reject_unknown_fields(payload, {"schema_version", "nvidia_smi"}, "PreprocessingRuntimeGpuEvidence")
    return PreprocessingRuntimeGpuEvidence(
        schema_version=_required_schema_version(payload, "PreprocessingRuntimeGpuEvidence"),
        nvidia_smi=_required_str(payload, "nvidia_smi"),
    )


def _optional_smoke_evidence(payload: Mapping[str, object]) -> PreprocessingRuntimeSmokeEvidence | None:
    value = _required_optional_mapping(payload, "smoke_evidence")
    return None if value is None else preprocessing_runtime_smoke_evidence_from_mapping(value)


def _validate_direct_schema_version(schema_version: int, record_name: str) -> None:
    validated = validate_schema_version(schema_version, record_name=record_name)
    if validated != schema_version:
        raise ValueError(f"{record_name} schema_version must be declared explicitly")


def _required_schema_version(payload: Mapping[str, object], record_name: str) -> int:
    if "schema_version" not in payload:
        raise ValueError(f"{record_name} is missing explicit schema_version")
    return validate_schema_version(payload["schema_version"], record_name=record_name)


def _validate_sha256(value: str, field_name: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")


def _validate_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _parse_timestamp(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"{field_name} must be an RFC 3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an RFC 3339 UTC timestamp") from exc
    return parsed


def _reject_unknown_fields(payload: Mapping[str, object], allowed: set[str], record_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {record_name} field(s): {', '.join(unknown)}")


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_optional_str(payload: Mapping[str, object], key: str) -> str | None:
    if key not in payload:
        raise ValueError(f"{key} is required")
    value = payload[key]
    if value is None:
        return None
    return _required_str(payload, key)


def _optional_str(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return _required_str(payload, key)


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_str_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings")
    return tuple(value)


def _required_optional_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    if key not in payload:
        raise ValueError(f"{key} is required")
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping or null")
    return value


__all__ = [
    "PREPROCESSING_ADAPTER_VERSION",
    "PREPROCESSING_RUNTIME_COMMAND",
    "PREPROCESSING_RUNTIME_CONTRACT_ID",
    "PreprocessingRuntimeGpuEvidence",
    "PreprocessingRuntimeImageEvidence",
    "PreprocessingRuntimeImageIdentity",
    "PreprocessingRuntimeQualificationRecord",
    "PreprocessingRuntimeQualificationTuple",
    "PreprocessingRuntimeSmokeEvidence",
    "PreprocessingRuntimeSourceEvidence",
    "PreprocessingRuntimeToolEvidence",
    "QualifiedPreprocessingRuntimeSelection",
    "normalize_rsync_version",
    "preprocessing_runtime_gpu_evidence_from_mapping",
    "preprocessing_runtime_image_evidence_from_mapping",
    "preprocessing_runtime_image_identity_from_mapping",
    "preprocessing_runtime_qualification_record_from_mapping",
    "preprocessing_runtime_qualification_tuple_from_mapping",
    "preprocessing_runtime_smoke_evidence_from_mapping",
    "preprocessing_runtime_source_evidence_from_mapping",
    "preprocessing_runtime_tool_evidence_from_mapping",
    "preprocessing_runtime_tuple_id",
    "qualified_preprocessing_runtime_selection_from_mapping",
    "validate_preprocessing_runtime_tool_evidence",
]
