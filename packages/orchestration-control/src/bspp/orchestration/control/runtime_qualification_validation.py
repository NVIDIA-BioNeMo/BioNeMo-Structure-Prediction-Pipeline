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

"""Phase-neutral validation for promoted Runtime Qualification authority."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.runtime_qualification import (
    RuntimeIpsaeEvidence,
    SelectedSourceIdentity,
    runtime_ipsae_evidence_from_mapping,
    selected_source_identity_from_mapping,
)
from bspp.orchestration.contract.source_package import (
    SourcePackageIdentity,
    source_package_identity_from_mapping,
)
from bspp.orchestration.control.execution_bootstrap import ImageIdentity, image_identity_from_mapping
from bspp.orchestration.control.profiles import ResolvedClusterProfile

RuntimeSourceKind = Literal["baked", "override"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ATTEMPT_TOKEN = re.compile(r"[0-9a-f]{32}")
_JOB_ID = re.compile(r"[0-9]+")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "profile",
    "status",
    "evidence_status",
    "submitted_at",
    "qualified_at",
    "expires_at",
    "tuple_id",
    "tuple",
    "attempt_history",
    "smoke_job",
    "smoke_evidence",
}
_TOP_LEVEL_OPTIONAL_FIELDS = {"autorequeue_cap"}
_SMOKE_FIELDS_V1 = {
    "tuple_id",
    "job_id",
    "status",
    "attempt_token",
    "python",
    "gpu",
    "bootstrap_sha256",
    "source_package_identity",
    "toolkit_package_identity",
    "image_identity",
    "selected_source_identity",
    "runtime_ipsae",
}
_SMOKE_FIELDS = _SMOKE_FIELDS_V1 | {"publication_compatibility"}
_SMOKE_JOB_FIELDS = {
    "job_id",
    "script_path",
    "submit_command",
    "record_path",
    "attempt_token",
    "input_path",
    "result_path",
    "remote_script_path",
    "remote_input_path",
    "remote_result_path",
}
_COMMON_TUPLE_FIELDS = {
    "cluster_profile",
    "scheduling_class",
    "execution_runtime_image",
    "runtime_image_policy",
    "runtime_image_size_bytes",
    "runtime_image_sha256",
    "image_identity",
    "runtime_facts",
    "source_bundle_id",
    "source_bundle_path",
    "source_package_identity",
    "selected_source",
}


@dataclass(frozen=True)
class RuntimeQualificationAutorequeueCap:
    """Qualified fixed autorequeue site cap recovered from a promoted record."""

    requeue_exit: int
    max_batch_requeue: int


@dataclass(frozen=True)
class ValidatedRuntimeQualification:
    """Authentic promoted-v1 facts recovered from one exact document."""

    profile_name: str
    tuple_id: str
    qualified_at: str
    expires_at: str
    source_kind: RuntimeSourceKind
    image: ImageIdentity
    source_package: SourcePackageIdentity
    toolkit_package: SourcePackageIdentity | None
    selected_source: SelectedSourceIdentity
    runtime_ipsae: RuntimeIpsaeEvidence
    remote_result_path: str
    bootstrap_sha256: str
    autorequeue_cap: RuntimeQualificationAutorequeueCap | None


def validate_promoted_runtime_qualification(
    document: bytes,
    *,
    profile: ResolvedClusterProfile | None,
    source_repo: Path | None,
    observed_at: datetime | None,
) -> ValidatedRuntimeQualification:
    """Validate exact canonical bytes as authentic promoted-v1 authority."""
    payload = _canonical_mapping(document)
    payload_keys = set(payload)
    allowed_keys = _TOP_LEVEL_FIELDS | _TOP_LEVEL_OPTIONAL_FIELDS
    if (
        not _TOP_LEVEL_FIELDS.issubset(payload_keys)
        or not payload_keys.issubset(allowed_keys)
        or payload.get("schema_version") != 1
    ):
        raise ValueError("Runtime Qualification is not an authentic promoted v1 record")
    if payload.get("status") != "qualified" or payload.get("evidence_status") != "qualified":
        raise ValueError("Runtime Qualification must be qualified with qualified evidence")
    profile_name = _string(payload, "profile")
    if profile is not None and profile_name != profile.name:
        raise ValueError("Runtime Qualification profile differs from the selected profile")
    _require_current_window(payload, observed_at=observed_at)
    attempt_history = payload.get("attempt_history")
    if not isinstance(attempt_history, list):
        raise ValueError("Runtime Qualification attempt_history must be a list")

    tuple_mapping = _mapping(payload, "tuple")
    source_kind = _source_kind(tuple_mapping)
    expected_tuple_fields = _COMMON_TUPLE_FIELDS | (
        {"toolkit_source", "toolkit_package_identity", "toolkit_package_path"} if source_kind == "override" else set()
    )
    if set(tuple_mapping) != expected_tuple_fields:
        raise ValueError("Runtime Qualification tuple has missing or extra fields")
    tuple_id = _string(payload, "tuple_id")
    if tuple_id != canonical_mapping_digest(tuple_mapping):
        raise ValueError("Runtime Qualification tuple_id differs from its exact tuple")

    image_mapping = _mapping(tuple_mapping, "image_identity")
    image = image_identity_from_mapping(image_mapping)
    source_mapping = _mapping(tuple_mapping, "source_package_identity")
    source_package = source_package_identity_from_mapping(source_mapping)
    if source_package.package_role != "orchestration":
        raise ValueError("Runtime Qualification source package role is not orchestration")
    selected_mapping = _mapping(tuple_mapping, "selected_source")
    selected = selected_source_identity_from_mapping(selected_mapping)
    if selected.source_kind != source_kind:
        raise ValueError("Runtime Qualification selected source kind is inconsistent")
    _validate_tuple_intrinsic(
        tuple_mapping,
        profile_name=profile_name,
        image=image,
        source_package=source_package,
    )

    toolkit_mapping: Mapping[str, object] | None = None
    toolkit_package: SourcePackageIdentity | None = None
    if source_kind == "override":
        toolkit_mapping = _mapping(tuple_mapping, "toolkit_package_identity")
        toolkit_package = source_package_identity_from_mapping(toolkit_mapping)
        if toolkit_package.package_role != "toolkit" or selected.toolkit_package_identity != dict(toolkit_mapping):
            raise ValueError("Runtime Qualification toolkit package binding is invalid")
        if tuple_mapping.get("toolkit_package_path") != str(toolkit_package.package_path):
            raise ValueError("Runtime Qualification toolkit package paths differ")
    elif selected.image_identity != dict(image_mapping):
        raise ValueError("Runtime Qualification baked source does not bind the qualified image")

    if profile is not None:
        _validate_tuple_profile(
            tuple_mapping,
            profile=profile,
            image=image,
            source_kind=source_kind,
            source_package=source_package,
        )
        _validate_configured_promoted_attempt_paths(
            _mapping(payload, "smoke_job"),
            profile=profile,
            tuple_id=tuple_id,
        )
    if source_repo is not None:
        validate_runtime_qualification_source(source_repo, commit=source_package.commit, tree=source_package.tree)

    smoke = _mapping(payload, "smoke_evidence")
    if set(smoke) not in (_SMOKE_FIELDS_V1, _SMOKE_FIELDS) or smoke.get("status") != "succeeded":
        raise ValueError("Runtime Qualification smoke evidence is not authentic")
    smoke_job = _mapping(payload, "smoke_job")
    if set(smoke_job) != _SMOKE_JOB_FIELDS:
        raise ValueError("Runtime Qualification smoke job has missing or extra fields")
    _validate_promoted_attempt_paths(
        smoke_job,
        profile_name=profile_name,
        tuple_id=tuple_id,
    )
    _validate_smoke_bindings(
        smoke,
        smoke_job=smoke_job,
        tuple_id=tuple_id,
        source_mapping=source_mapping,
        toolkit_mapping=toolkit_mapping,
        image_mapping=image_mapping,
        selected_mapping=selected_mapping,
    )
    runtime_ipsae = runtime_ipsae_evidence_from_mapping(_mapping(smoke, "runtime_ipsae"))
    if source_kind == "baked":
        # Reconstruct the validated selected_source from the smoke's reported
        # revision (the image's actual baked commit, bound by the image sha256);
        # the tuple's selected_source.revision is only a public placeholder.
        smoke_selected = selected_source_identity_from_mapping(_mapping(smoke, "selected_source_identity"))
        selected = SelectedSourceIdentity(
            source_kind=selected.source_kind,
            root=selected.root,
            revision=smoke_selected.revision,
            image_identity=selected.image_identity,
            toolkit_package_identity=None,
        )
    if runtime_ipsae.source_revision != selected.revision:
        raise ValueError("Runtime Qualification iPSAE revision differs from the selected toolkit source")
    if "publication_compatibility" in smoke:
        _validate_publication_compatibility(_mapping(smoke, "publication_compatibility"))
    return ValidatedRuntimeQualification(
        profile_name=profile_name,
        tuple_id=tuple_id,
        qualified_at=_string(payload, "qualified_at"),
        expires_at=_string(payload, "expires_at"),
        source_kind=source_kind,
        image=image,
        source_package=source_package,
        toolkit_package=toolkit_package,
        selected_source=selected,
        runtime_ipsae=runtime_ipsae,
        remote_result_path=_absolute_string(smoke_job, "remote_result_path"),
        bootstrap_sha256=_sha_string(smoke, "bootstrap_sha256"),
        autorequeue_cap=_autorequeue_cap(payload),
    )


def _autorequeue_cap(payload: Mapping[str, object]) -> RuntimeQualificationAutorequeueCap | None:
    value = payload.get("autorequeue_cap")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Runtime Qualification autorequeue_cap must be a mapping")
    cap = cast("Mapping[str, object]", value)
    if set(cap) != {"requeue_exit", "max_batch_requeue"}:
        raise ValueError("Runtime Qualification autorequeue_cap must have exactly requeue_exit and max_batch_requeue")
    requeue_exit = cap.get("requeue_exit")
    max_batch_requeue = cap.get("max_batch_requeue")
    if not isinstance(requeue_exit, int) or isinstance(requeue_exit, bool) or requeue_exit != 85:
        raise ValueError("Runtime Qualification autorequeue_cap requeue_exit must equal 85")
    if not isinstance(max_batch_requeue, int) or isinstance(max_batch_requeue, bool) or max_batch_requeue < 0:
        raise ValueError("Runtime Qualification autorequeue_cap max_batch_requeue must be a non-negative integer")
    return RuntimeQualificationAutorequeueCap(requeue_exit=requeue_exit, max_batch_requeue=max_batch_requeue)


def _validate_publication_compatibility(value: Mapping[str, object]) -> None:
    if set(value) != {
        "schema_version",
        "check",
        "status",
        "process_count",
        "published_directory_count",
        "fallback_errno",
        "output_identity",
        "artifact",
    }:
        raise ValueError("Runtime Qualification publication compatibility evidence has missing or extra fields")
    output = _mapping(value, "output_identity")
    artifact = _mapping(value, "artifact")
    if (
        value.get("schema_version") != 1
        or value.get("check") != "postprocessing-directory-publication-v1"
        or value.get("status") != "passed"
        or value.get("process_count") != 2
        or value.get("published_directory_count") != 1
        or value.get("fallback_errno") != 22
        or set(output) != {"path", "sha256", "size_bytes"}
        or output.get("path") != "published/payload.json"
        or not isinstance(output.get("sha256"), str)
        or _SHA256.fullmatch(cast(str, output.get("sha256"))) is None
        or type(output.get("size_bytes")) is not int
        or set(artifact) != {"path", "sha256", "size_bytes"}
        or artifact.get("path") != "publication-compatibility.json"
        or not isinstance(artifact.get("sha256"), str)
        or _SHA256.fullmatch(cast(str, artifact.get("sha256"))) is None
        or type(artifact.get("size_bytes")) is not int
        or not 0 < cast(int, artifact.get("size_bytes")) <= 4096
    ):
        raise ValueError("Runtime Qualification publication compatibility evidence is invalid")


def expected_runtime_qualification_attempt_paths(
    *,
    profile: ResolvedClusterProfile,
    tuple_id: str,
    attempt_token: str,
) -> dict[str, Path]:
    """Return the exact controller/cluster paths bound to an attempt token."""
    control_root = profile.runtime_qualification_control_root
    remote_root = profile.runtime_qualification_root
    if control_root is None or remote_root is None:
        raise ValueError("Runtime Qualification roots are not configured")
    local_attempt = Path(control_root) / profile.name / "attempts" / tuple_id / attempt_token
    remote_attempt = (
        Path(remote_root) / profile.name / "attempts" / tuple_id / attempt_token
        if profile.transport == "ssh"
        else local_attempt
    )
    return {
        "script_path": local_attempt / "smoke.sbatch",
        "input_path": local_attempt / "input.json",
        "result_path": local_attempt / "result.json",
        "remote_script_path": remote_attempt / "smoke.sbatch",
        "remote_input_path": remote_attempt / "input.json",
        "remote_result_path": remote_attempt / "result.json",
    }


def validate_runtime_qualification_attempt_binding(
    payload: Mapping[str, object],
    *,
    profile: ResolvedClusterProfile,
    tuple_id: str,
    record_path: Path,
) -> None:
    """Validate persisted current-attempt paths against configured ancestry."""
    smoke_job = payload.get("smoke_job")
    if not isinstance(smoke_job, Mapping):
        raise ValueError("Runtime Qualification record lacks smoke job binding")
    token = smoke_job.get("attempt_token")
    if not isinstance(token, str) or _ATTEMPT_TOKEN.fullmatch(token) is None:
        raise ValueError("Runtime Qualification attempt token is invalid")
    expected = expected_runtime_qualification_attempt_paths(
        profile=profile,
        tuple_id=tuple_id,
        attempt_token=token,
    )
    if smoke_job.get("record_path") != str(record_path):
        raise ValueError("Runtime Qualification record path binding is invalid")
    for name, expected_path in expected.items():
        if smoke_job.get(name) != str(expected_path):
            raise ValueError(f"Runtime Qualification {name} binding is invalid")


def _validate_configured_promoted_attempt_paths(
    smoke_job: Mapping[str, object],
    *,
    profile: ResolvedClusterProfile,
    tuple_id: str,
) -> None:
    """Bind whichever profile-owned attempt roots are available."""
    token = _string(smoke_job, "attempt_token")
    control_root = profile.runtime_qualification_control_root
    if control_root is not None:
        local_attempt = Path(control_root) / profile.name / "attempts" / tuple_id / token
        expected_local = {
            "record_path": Path(control_root) / profile.name / f"{tuple_id}.json",
            "script_path": local_attempt / "smoke.sbatch",
            "input_path": local_attempt / "input.json",
            "result_path": local_attempt / "result.json",
        }
        for name, expected in expected_local.items():
            if smoke_job.get(name) != str(expected):
                raise ValueError(f"Runtime Qualification {name} differs from the configured control root")
    remote_root = profile.runtime_qualification_root
    if remote_root is not None:
        remote_attempt = Path(remote_root) / profile.name / "attempts" / tuple_id / token
        expected_remote = {
            "remote_script_path": remote_attempt / "smoke.sbatch",
            "remote_input_path": remote_attempt / "input.json",
            "remote_result_path": remote_attempt / "result.json",
        }
        for name, expected in expected_remote.items():
            if smoke_job.get(name) != str(expected):
                raise ValueError(f"Runtime Qualification {name} differs from the configured Runtime root")


def _validate_tuple_profile(
    payload: Mapping[str, object],
    *,
    profile: ResolvedClusterProfile,
    image: ImageIdentity,
    source_kind: RuntimeSourceKind,
    source_package: SourcePackageIdentity,
) -> None:
    worker = profile.resources.get("gpu_worker")
    runtime_facts = payload.get("runtime_facts")
    expected_kind: RuntimeSourceKind = "baked" if not profile.afdb_toolkit_repo else "override"
    if (
        payload.get("cluster_profile") != profile.name
        or payload.get("scheduling_class") != "gpu_worker"
        or payload.get("execution_runtime_image") != str(image.path)
        or payload.get("runtime_image_policy") != image.policy
        or payload.get("runtime_image_size_bytes") != image.size_bytes
        or payload.get("runtime_image_sha256") != image.sha256
        or payload.get("source_bundle_path") != str(source_package.package_path)
        or source_kind != expected_kind
        or not isinstance(runtime_facts, Mapping)
        or set(runtime_facts) != {"gpu_worker_gres"}
        or runtime_facts.get("gpu_worker_gres") != (worker.gres if worker is not None else None)
    ):
        raise ValueError("Runtime Qualification tuple differs from selected profile/runtime facts")
    if source_kind == "override" and payload.get("toolkit_source") != profile.afdb_toolkit_repo:
        raise ValueError("Runtime Qualification toolkit source differs from the selected profile")


def _validate_tuple_intrinsic(
    payload: Mapping[str, object],
    *,
    profile_name: str,
    image: ImageIdentity,
    source_package: SourcePackageIdentity,
) -> None:
    runtime_facts = payload.get("runtime_facts")
    source_bundle_id = f"bspp-orchestration-{source_package.commit}"
    legacy_source_bundle_id = f"afcdb-orchestration-{source_package.commit}"
    if (
        payload.get("cluster_profile") != profile_name
        or payload.get("scheduling_class") != "gpu_worker"
        or payload.get("execution_runtime_image") != str(image.path)
        or payload.get("runtime_image_policy") != image.policy
        or payload.get("runtime_image_size_bytes") != image.size_bytes
        or payload.get("runtime_image_sha256") != image.sha256
        or payload.get("source_bundle_id") not in (source_bundle_id, legacy_source_bundle_id)
        or payload.get("source_bundle_path") != str(source_package.package_path)
        or not isinstance(runtime_facts, Mapping)
        or set(runtime_facts) != {"gpu_worker_gres"}
        or (
            runtime_facts.get("gpu_worker_gres") is not None
            and not isinstance(runtime_facts.get("gpu_worker_gres"), str)
        )
    ):
        raise ValueError("Runtime Qualification tuple has inconsistent intrinsic bindings")


def _selected_smoke_binds(selected_smoke: SelectedSourceIdentity, selected_mapping: Mapping[str, object]) -> bool:
    """Validate the smoke's selected_source_identity against the tuple, relaxing
    only the baked-mode revision (the image's actual baked commit, bound by the
    image sha256, is authoritative and cannot be known a priori)."""
    if selected_smoke.source_kind != selected_mapping.get("source_kind"):
        return False
    if selected_smoke.root != selected_mapping.get("root"):
        return False
    if selected_smoke.source_kind == "baked":
        return (
            selected_smoke.image_identity == selected_mapping.get("image_identity")
            and _COMMIT.fullmatch(selected_smoke.revision) is not None
        )
    return selected_smoke.toolkit_package_identity == selected_mapping.get(
        "toolkit_package_identity"
    ) and selected_smoke.revision == selected_mapping.get("revision")


def _validate_smoke_bindings(
    smoke: Mapping[str, object],
    *,
    smoke_job: Mapping[str, object],
    tuple_id: str,
    source_mapping: Mapping[str, object],
    toolkit_mapping: Mapping[str, object] | None,
    image_mapping: Mapping[str, object],
    selected_mapping: Mapping[str, object],
) -> None:
    job_id = _string(smoke, "job_id")
    token = _string(smoke, "attempt_token")
    selected_smoke = selected_source_identity_from_mapping(_mapping(smoke, "selected_source_identity"))
    submit_command = smoke_job.get("submit_command")
    if (
        _JOB_ID.fullmatch(job_id) is None
        or _ATTEMPT_TOKEN.fullmatch(token) is None
        or smoke_job.get("job_id") != job_id
        or smoke_job.get("attempt_token") != token
        or not isinstance(submit_command, list)
        or not submit_command
        or any(not isinstance(item, str) or not item or "\x00" in item for item in submit_command)
        or smoke.get("tuple_id") != tuple_id
        or smoke.get("source_package_identity") != source_mapping
        or smoke.get("toolkit_package_identity") != toolkit_mapping
        or smoke.get("image_identity") != image_mapping
        or not _selected_smoke_binds(selected_smoke, selected_mapping)
        or not _string(smoke, "python")
        or not _string(smoke, "gpu")
    ):
        raise ValueError("Runtime Qualification smoke evidence differs from its tuple/job")


def _validate_promoted_attempt_paths(
    smoke_job: Mapping[str, object],
    *,
    profile_name: str,
    tuple_id: str,
) -> None:
    """Validate exact attempt topology without relying on ambient profile roots."""
    token = _string(smoke_job, "attempt_token")
    if _ATTEMPT_TOKEN.fullmatch(token) is None:
        raise ValueError("Runtime Qualification attempt token is invalid")
    paths = {
        name: _absolute_path(smoke_job, name)
        for name in (
            "record_path",
            "script_path",
            "input_path",
            "result_path",
            "remote_script_path",
            "remote_input_path",
            "remote_result_path",
        )
    }
    record_path = paths["record_path"]
    if record_path.name != f"{tuple_id}.json" or record_path.parent.name != profile_name:
        raise ValueError("Runtime Qualification record path has invalid intrinsic identity")
    # Promoted-v1 authority exists from both producer layouts: the original
    # control-root/attempts form and the current profile-scoped form. Replay
    # accepts only those two exact intrinsic layouts; an ambient profile below
    # additionally restricts live use to its currently configured roots.
    current_local_attempt = record_path.parent / "attempts" / tuple_id / token
    historical_local_attempt = record_path.parent.parent / "attempts" / tuple_id / token
    local_attempt: Path | None = None
    for candidate in (current_local_attempt, historical_local_attempt):
        if all(
            paths[name] == candidate / basename
            for name, basename in (
                ("script_path", "smoke.sbatch"),
                ("input_path", "input.json"),
                ("result_path", "result.json"),
            )
        ):
            local_attempt = candidate
            break
    if local_attempt is None:
        raise ValueError("Runtime Qualification local paths have invalid intrinsic attempt binding")
    remote_attempt = paths["remote_script_path"].parent
    expected_remote_tail = (
        (profile_name, "attempts", tuple_id, token)
        if local_attempt == current_local_attempt
        else ("attempts", tuple_id, token)
    )
    if remote_attempt.parts[-len(expected_remote_tail) :] != expected_remote_tail:
        raise ValueError("Runtime Qualification remote attempt path has invalid intrinsic identity")
    for name, basename in (
        ("remote_script_path", "smoke.sbatch"),
        ("remote_input_path", "input.json"),
        ("remote_result_path", "result.json"),
    ):
        if paths[name] != remote_attempt / basename:
            raise ValueError(f"Runtime Qualification {name} has invalid intrinsic attempt binding")


def validate_runtime_qualification_source(source_repo: Path, *, commit: str, tree: str) -> None:
    root = source_repo.resolve(strict=True)
    observed_commit = _git(root, "rev-parse", "HEAD")
    observed_tree = _git(root, "rev-parse", "HEAD^{tree}")
    status_text = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status_text or (observed_commit, observed_tree) != (commit, tree):
        raise ValueError("Runtime Qualification differs from the current clean source checkout")


def _require_current_window(payload: Mapping[str, object], *, observed_at: datetime | None) -> None:
    qualified = _timestamp(payload.get("qualified_at"), "qualified_at")
    expires = _timestamp(payload.get("expires_at"), "expires_at")
    submitted = _timestamp(payload.get("submitted_at"), "submitted_at")
    if not submitted <= qualified < expires:
        raise ValueError("Runtime Qualification timestamp order is invalid")
    if observed_at is None:
        return
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("Runtime Qualification observation time must be timezone-aware")
    moment = observed_at.astimezone(UTC)
    if moment < qualified or moment >= expires:
        raise ValueError("Runtime Qualification is not current")


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"Runtime Qualification {label} must be canonical UTC")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"Runtime Qualification {label} is invalid") from exc


def _canonical_mapping(document: bytes) -> Mapping[str, object]:
    try:
        payload = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Runtime Qualification must be UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Runtime Qualification must be a JSON mapping")
    canonical = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if document != canonical:
        raise ValueError("Runtime Qualification must use producer-canonical JSON bytes")
    return cast("Mapping[str, object]", payload)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(repo), *args), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "Runtime Qualification source identity git command failed")
    return result.stdout.strip()


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Runtime Qualification {key} must be a mapping")
    return cast("Mapping[str, object]", value)


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Runtime Qualification {key} must be non-empty")
    return value


def _absolute_string(payload: Mapping[str, object], key: str) -> str:
    return str(_absolute_path(payload, key))


def _absolute_path(payload: Mapping[str, object], key: str) -> Path:
    value = _string(payload, key)
    path = Path(value)
    if "\x00" in value or "\n" in value or not path.is_absolute() or str(path) != value or ".." in path.parts:
        raise ValueError(f"Runtime Qualification {key} must be a normalized absolute path")
    return path


def _sha_string(payload: Mapping[str, object], key: str) -> str:
    value = _string(payload, key)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"Runtime Qualification {key} must be SHA-256")
    return value


def _source_kind(payload: Mapping[str, object]) -> RuntimeSourceKind:
    selected = _mapping(payload, "selected_source")
    kind = selected.get("source_kind")
    if kind not in {"baked", "override"}:
        raise ValueError("Runtime Qualification source kind is invalid")
    revision = selected.get("revision")
    if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
        raise ValueError("Runtime Qualification source revision is invalid")
    return cast("RuntimeSourceKind", kind)


__all__ = [
    "RuntimeQualificationAutorequeueCap",
    "RuntimeSourceKind",
    "ValidatedRuntimeQualification",
    "expected_runtime_qualification_attempt_paths",
    "validate_promoted_runtime_qualification",
    "validate_runtime_qualification_attempt_binding",
    "validate_runtime_qualification_source",
]
