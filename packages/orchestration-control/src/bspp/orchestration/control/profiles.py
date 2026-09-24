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

"""Cluster Profile resolution for Run Plan expansion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from bspp.orchestration.contract.config_models import FrozenConfigModel
from bspp.orchestration.contract.database_cache_maintenance import (
    DatabaseAcceptanceCacheProfile,
    reject_profile_inheritance,
    select_database_acceptance_cache_profile,
)
from bspp.orchestration.contract.database_placement import (
    DatabaseAccessPolicy,
    DatabaseProfileStagingSnapshot,
    DatabaseSetSelection,
)
from bspp.orchestration.contract.database_set_provisioning import DatabaseSetIdentity
from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot
from bspp.orchestration.contract.folding_release import (
    FoldingBackendImageOverride,
    FoldingReleasePreset,
)
from bspp.orchestration.contract.postprocessing_execution import PostprocessingCredentialMountSnapshot
from bspp.orchestration.contract.preprocessing_runtime import normalize_rsync_version
from bspp.orchestration.contract.runplan import reject_environment_interpolation
from bspp.orchestration.contract.runspec import SlurmResources
from bspp.orchestration.contract.runspec_policies import parse_slurm_time_seconds

_TEMPLATE_RESOURCE = files("bspp.orchestration.control.data").joinpath("cluster_profile_templates.yaml")
TransportKind = Literal["ssh", "local-slurm"]
RuntimeImagePolicy = Literal["digest-checked", "trusted-cache"]
_PATH_KEYS = (
    "project_root",
    "output_root",
    "staging_root",
    "afdb_toolkit_repo",
    "orchestration_repo",
    "image",
    "probe_root",
    "source_bundle_root",
    "runtime_image_cache_root",
    "runtime_qualification_root",
    "runtime_qualification_control_root",
    "governed_package_root",
)
_DATABASE_PROFILE_KEYS = frozenset(
    {
        "database_sets",
        "database_access_policies",
        "database_cache_root",
        "database_cache_namespace",
        "database_cache_unix_user",
        "database_cache_filesystem_type",
        "database_cache_reserve_bytes",
        "database_lock_wait_seconds",
    }
)


class ProfileMount(FrozenConfigModel):
    """Container mount requested by a user-local Cluster Profile."""

    source: StrictStr
    target: StrictStr
    read_only: bool = False

    @field_validator("source", "target")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        if value == "":
            msg = "Expected non-empty string"
            raise ValueError(msg)
        return value

    @model_serializer(mode="wrap")
    def _omit_default_read_only(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """Omit ``read_only`` when false so legacy projections stay byte-stable.

        The legacy workflow RunSpec contract removed ``container.mounts[].read_only``
        and rejects the key outright, so a default mount must serialize exactly as
        before the read_only removal. An explicit ``read_only: true`` still serializes, and the
        legacy loader then fails closed — a flow that cannot express read-only must
        not silently downgrade it to read-write.
        """
        data: dict[str, object] = handler(self)
        if not self.read_only:
            data.pop("read_only", None)
        return data


class ClusterProfileTemplate(FrozenConfigModel):
    """Repo-provided shared defaults for one cluster."""

    account: StrictStr
    resources: Mapping[str, SlurmResources] = Field(default_factory=dict)

    @field_validator("account")
    @classmethod
    def _validate_non_empty_account(cls, value: str) -> str:
        if value == "":
            msg = "Expected non-empty account"
            raise ValueError(msg)
        return value


class PreprocessingRuntimeProfile(FrozenConfigModel):
    """Immutable preprocessing image selection expected by qualification."""

    cluster_image_path: StrictStr
    cluster_image_sha256: StrictStr
    oci_digest: StrictStr
    image_lock_sha256: StrictStr
    contract_wheel_sha256: StrictStr
    runtime_wheel_sha256: StrictStr
    control_wheel_sha256: StrictStr | None = None
    source_commit: StrictStr
    source_bundle_sha256: StrictStr
    colabfold_version: StrictStr
    mmseqs_version: StrictStr
    rsync_version: StrictStr
    cuda_version: StrictStr

    @field_validator("cluster_image_path")
    @classmethod
    def _validate_image_path(cls, value: str) -> str:
        if not value:
            raise ValueError("preprocessing runtime cluster_image_path must be non-empty")
        return value

    @field_validator(
        "cluster_image_sha256",
        "image_lock_sha256",
        "contract_wheel_sha256",
        "runtime_wheel_sha256",
        "source_bundle_sha256",
    )
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("preprocessing runtime SHA-256 fields must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("control_wheel_sha256")
    @classmethod
    def _validate_control_wheel_sha256(cls, value: str | None) -> str | None:
        # Optional: pre-Control images and legacy 12-field profiles omit it.
        if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("preprocessing runtime control_wheel_sha256 must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("oci_digest")
    @classmethod
    def _validate_oci_digest(cls, value: str) -> str:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ValueError("preprocessing runtime oci_digest must be sha256:<64 lowercase hex>")
        return value

    @field_validator("source_commit")
    @classmethod
    def _validate_source_commit(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("preprocessing runtime source_commit must be a full lowercase Git SHA-1")
        return value

    @field_validator("colabfold_version", "mmseqs_version", "cuda_version")
    @classmethod
    def _validate_tool_version(cls, value: str) -> str:
        if not value:
            raise ValueError("preprocessing runtime tool versions must be non-empty")
        return value

    @field_validator("rsync_version")
    @classmethod
    def _validate_rsync_version(cls, value: str) -> str:
        normalized = normalize_rsync_version(value)
        if normalized != value:
            raise ValueError("preprocessing runtime rsync_version must be normalized")
        return value


class DatabaseSetManifestProfile(FrozenConfigModel):
    """Strict site mapping from one logical Database Set to one manifest."""

    identifier: StrictStr
    version: StrictStr
    manifest_path: StrictStr

    @field_validator("identifier", "version")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        DatabaseSetIdentity(identifier=value, version="validation")
        return value

    @field_validator("manifest_path")
    @classmethod
    def _validate_manifest_path(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("database manifest_path must be absolute")
        return value


class ClusterProfilePaths(FrozenConfigModel):
    """Filesystem and image locations in a user-local Cluster Profile."""

    project_root: StrictStr
    output_root: StrictStr
    staging_root: StrictStr
    afdb_toolkit_repo: StrictStr | None = None
    orchestration_repo: StrictStr
    image: StrictStr
    probe_root: StrictStr | None = None
    source_bundle_root: StrictStr | None = None
    runtime_image_cache_root: StrictStr | None = None
    runtime_qualification_root: StrictStr | None = None
    runtime_qualification_control_root: StrictStr | None = None
    governed_package_root: StrictStr | None = None

    @field_validator(*_PATH_KEYS)
    @classmethod
    def _validate_non_empty_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            msg = "Expected non-empty string"
            raise ValueError(msg)
        return value


class PostprocessingCredentialMountProfile(FrozenConfigModel):
    """Cluster-visible AWS shared-profile files used by postprocessing jobs."""

    aws_shared_credentials_file: StrictStr
    aws_config_file: StrictStr

    @model_validator(mode="after")
    def _validate_mount_sources(self) -> Self:
        PostprocessingCredentialMountSnapshot(
            aws_shared_credentials_file=self.aws_shared_credentials_file,
            aws_config_file=self.aws_config_file,
        )
        return self


class FoldingBackendAssetsProfile(FrozenConfigModel):
    """One unique backend-tagged external-asset record in a Cluster Profile.

    The closed ``FoldingBackendAssetsSnapshot`` contract is constructed in the
    ``after`` validator so its rules are inherited here: backend membership,
    per-backend required/disallowed fields, absolute/empty path rejection, and
    the ``.pt`` suffix rule for BioIR. ``extra="forbid"`` rejects unknown keys,
    including any TRT ``model_fn`` factory field.
    """

    backend: StrictStr
    chain_manifest_csv: StrictStr | None = None
    openfold_model_dir: StrictStr | None = None
    colabfold_weights_dir: StrictStr | None = None
    bioir_checkpoint: StrictStr | None = None
    bioir_monomer_checkpoint: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_snapshot(self) -> Self:
        self.to_snapshot()
        return self

    def to_snapshot(self) -> FoldingBackendAssetsSnapshot:
        """Return the immutable contract snapshot for this profile record."""
        return FoldingBackendAssetsSnapshot(
            backend=self.backend,
            chain_manifest_csv=self.chain_manifest_csv,
            openfold_model_dir=self.openfold_model_dir,
            colabfold_weights_dir=self.colabfold_weights_dir,
            bioir_checkpoint=self.bioir_checkpoint,
            bioir_monomer_checkpoint=self.bioir_monomer_checkpoint,
        )


class UserClusterProfile(FrozenConfigModel):
    """User-local cluster paths and override knobs."""

    owner: StrictStr
    project_root: StrictStr
    output_root: StrictStr
    staging_root: StrictStr
    afdb_toolkit_repo: StrictStr | None = None
    orchestration_repo: StrictStr
    image: StrictStr
    transport: TransportKind
    ssh_target: StrictStr | None = None
    probe_root: StrictStr | None = None
    source_bundle_root: StrictStr | None = None
    runtime_image_cache_root: StrictStr | None = None
    runtime_image_policy: RuntimeImagePolicy = "digest-checked"
    runtime_qualification_root: StrictStr | None = None
    runtime_qualification_control_root: StrictStr | None = None
    governed_package_root: StrictStr | None = None
    runtime_qualification_expires_hours: int = 168
    preprocessing_runtime: PreprocessingRuntimeProfile | None = None
    postprocessing_credential_mounts: PostprocessingCredentialMountProfile | None = None
    database_sets: tuple[DatabaseSetManifestProfile, ...] = ()
    database_access_policies: tuple[DatabaseAccessPolicy, ...] = (DatabaseAccessPolicy.STAGE_REQUIRED,)
    database_cache_root: StrictStr | None = None
    database_cache_namespace: Literal["acceptance"] | None = None
    database_cache_unix_user: StrictStr | None = None
    database_cache_filesystem_type: StrictStr | None = None
    database_cache_reserve_bytes: StrictInt | None = None
    database_lock_wait_seconds: StrictInt | None = None
    extra_mounts: tuple[ProfileMount, ...] = ()
    folding_release_preset: str | None = None
    folding_backend_images: tuple[FoldingBackendImageOverride, ...] = ()
    folding_backend_assets: tuple[FoldingBackendAssetsProfile, ...] = ()
    mount_orchestration_source: bool = False
    account: StrictStr | None = None
    resources: Mapping[str, Mapping[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _promote_nested_paths(cls, value: object) -> object:
        if not isinstance(value, Mapping) or "paths" not in value:
            return value
        raw = dict(value)
        paths_value = raw.pop("paths")
        paths = ClusterProfilePaths.model_validate(paths_value).model_dump(mode="json", exclude_none=True)
        for key, path_value in paths.items():
            existing = raw.get(key)
            if existing is not None and existing != path_value:
                msg = f"Cluster Profile field {key!r} cannot be set both flat and under 'paths' with different values"
                raise ValueError(msg)
            raw[key] = path_value
        return raw

    @field_validator(
        "owner",
        "project_root",
        "output_root",
        "staging_root",
        "afdb_toolkit_repo",
        "orchestration_repo",
        "image",
        "ssh_target",
        "probe_root",
        "source_bundle_root",
        "runtime_image_cache_root",
        "runtime_qualification_root",
        "runtime_qualification_control_root",
        "database_cache_root",
        "database_cache_unix_user",
        "database_cache_filesystem_type",
        "governed_package_root",
    )
    @classmethod
    def _validate_non_empty_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            msg = "Expected non-empty string"
            raise ValueError(msg)
        return value

    @field_validator("runtime_qualification_expires_hours")
    @classmethod
    def _validate_positive_hours(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            msg = "runtime_qualification_expires_hours must be positive"
            raise ValueError(msg)
        return value

    @field_validator("database_cache_unix_user")
    @classmethod
    def _validate_cache_unix_user(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", value) is None:
            raise ValueError("database_cache_unix_user must be a safe Unix user identity")
        return value

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> Self:
        if self.transport == "ssh" and self.ssh_target is None:
            msg = "ssh transport requires ssh_target"
            raise ValueError(msg)
        if self.transport == "local-slurm" and self.ssh_target is not None:
            msg = "local-slurm transport must not set ssh_target"
            raise ValueError(msg)
        if (
            self.runtime_qualification_control_root is not None
            and not Path(self.runtime_qualification_control_root).is_absolute()
        ):
            raise ValueError("runtime_qualification_control_root must be absolute")
        identities = [(item.identifier, item.version) for item in self.database_sets]
        if len(identities) != len(set(identities)):
            raise ValueError("Cluster Profile database_sets must not contain duplicate identities")
        if not self.database_access_policies or len(self.database_access_policies) != len(
            set(self.database_access_policies)
        ):
            raise ValueError("Cluster Profile database_access_policies must be non-empty and unique")
        staging_capable = any(
            policy in {DatabaseAccessPolicy.STAGE_REQUIRED, DatabaseAccessPolicy.STAGE_PREFERRED}
            for policy in self.database_access_policies
        )
        database_configuration_authored = bool(self.model_fields_set & _DATABASE_PROFILE_KEYS)
        staging_values = (
            self.database_cache_root,
            self.database_cache_unix_user,
            self.database_cache_filesystem_type,
            self.database_cache_reserve_bytes,
            self.database_lock_wait_seconds,
        )
        if database_configuration_authored and staging_capable and any(value is None for value in staging_values):
            raise ValueError(
                "staging-capable Cluster Profile requires all database cache fields: "
                "database_cache_root, database_cache_unix_user, database_cache_filesystem_type, "
                "database_cache_reserve_bytes, database_lock_wait_seconds"
            )
        if any(value is not None for value in staging_values) and any(value is None for value in staging_values):
            raise ValueError("Cluster Profile requires all database cache fields when any are supplied")
        if self.database_cache_reserve_bytes is not None and self.database_cache_reserve_bytes < 0:
            raise ValueError("database_cache_reserve_bytes must be non-negative")
        if self.database_lock_wait_seconds is not None and self.database_lock_wait_seconds <= 0:
            raise ValueError("database_lock_wait_seconds must be positive")
        if self.folding_backend_images:
            backends = [entry.backend for entry in self.folding_backend_images]
            if len(backends) != len(set(backends)):
                raise ValueError("folding_backend_images must not contain duplicate backends")
        if self.folding_backend_assets:
            backends = [entry.backend for entry in self.folding_backend_assets]
            if len(backends) != len(set(backends)):
                raise ValueError("folding_backend_assets must not contain duplicate backends")
        return self


@dataclass(frozen=True)
class ResolvedClusterProfile:
    """Concrete cluster profile after applying user overrides to repo defaults."""

    name: str
    owner: str
    project_root: str
    output_root: str
    staging_root: str
    afdb_toolkit_repo: str | None
    orchestration_repo: str
    image: str
    transport: TransportKind
    ssh_target: str | None
    probe_root: str | None
    source_bundle_root: str | None
    runtime_image_cache_root: str | None
    runtime_image_policy: RuntimeImagePolicy
    runtime_qualification_root: str | None
    runtime_qualification_control_root: str | None
    governed_package_root: str | None
    runtime_qualification_expires_hours: int
    preprocessing_runtime: PreprocessingRuntimeProfile | None
    postprocessing_credential_mounts: PostprocessingCredentialMountProfile | None
    database_sets: tuple[DatabaseSetManifestProfile, ...]
    database_access_policies: tuple[DatabaseAccessPolicy, ...]
    database_staging: DatabaseProfileStagingSnapshot | None
    database_acceptance_cache: DatabaseAcceptanceCacheProfile | None
    extra_mounts: tuple[ProfileMount, ...]
    account: str
    resources: Mapping[str, SlurmResources]
    folding_release_preset: FoldingReleasePreset = FoldingReleasePreset.PUBLIC
    folding_backend_images: Mapping[str, str] = field(default_factory=dict)
    folding_backend_assets: Mapping[str, FoldingBackendAssetsSnapshot] = field(default_factory=dict)
    mount_orchestration_source: bool = False


def resolve_cluster_profile(
    cluster_name: str,
    *,
    config_path: Path,
    template_path: Path | None = None,
    config_document: bytes | None = None,
) -> ResolvedClusterProfile:
    """Resolve one named Cluster Profile from repo defaults plus user-local config."""
    template_clusters = _load_clusters(template_path)
    user_clusters = _load_clusters(config_path, document=config_document)
    missing_sources: list[str] = []
    raw_template: Mapping[str, object] = {}
    template_is_authority = False
    if cluster_name in template_clusters:
        raw_template = _selected_cluster_mapping(template_clusters, cluster_name)
        template_is_authority = True
    elif _is_example_name(cluster_name):
        generic_shape = _generic_template_shape(template_clusters)
        if isinstance(generic_shape, Mapping):
            raw_template = generic_shape
            template_is_authority = True
        else:
            missing_sources.append("repo Cluster Profile Template")
    if cluster_name not in user_clusters:
        missing_sources.append("user Cluster Profile")
    if missing_sources:
        msg = f"Unknown cluster profile {cluster_name!r}; missing from {', '.join(missing_sources)}"
        raise ValueError(msg)

    raw_user = _selected_cluster_mapping(user_clusters, cluster_name)
    reject_profile_inheritance(raw_user, source="user Cluster Profile")
    reject_environment_interpolation(raw_user, context="user Cluster Profile")

    user = UserClusterProfile.model_validate(raw_user)

    if template_is_authority:
        reject_profile_inheritance(raw_template, source="repo Cluster Profile Template")
        reject_environment_interpolation(raw_template, context="repo Cluster Profile Template")
        template = ClusterProfileTemplate.model_validate(raw_template)
        resources = _merge_resources(template.resources, user.resources)
        account = user.account or template.account
    else:
        # Non-template, non-example names never inherit the synthetic example
        # shape. The user mapping is the sole authority and must fully author a
        # non-empty account and complete resources without inherited fields.
        if not user.account:
            msg = (
                f"Unknown cluster profile {cluster_name!r}; missing from repo Cluster Profile Template "
                "and the user Cluster Profile does not author a non-empty account"
            )
            raise ValueError(msg)
        resources = _merge_resources({}, user.resources)
        if not resources:
            msg = (
                f"Unknown cluster profile {cluster_name!r}; missing from repo Cluster Profile Template "
                "and the user Cluster Profile does not author any resources"
            )
            raise ValueError(msg)
        account = user.account
    control_root = user.runtime_qualification_control_root
    if user.transport == "ssh" and user.runtime_qualification_root is not None and control_root is None:
        raise ValueError("SSH Cluster Profile requires runtime_qualification_control_root")
    if user.transport == "local-slurm" and control_root is None:
        control_root = user.runtime_qualification_root
    staging = _database_staging_snapshot(user)
    acceptance_cache = (
        select_database_acceptance_cache_profile({"clusters": user_clusters}, cluster_name)
        if user.database_cache_namespace == "acceptance"
        else None
    )
    if staging is not None:
        try:
            wall_time = resources["gpu_worker"].time
        except KeyError as exc:
            raise ValueError("staging-capable Cluster Profile requires gpu_worker resources") from exc
        if staging.lock_wait_seconds > parse_slurm_time_seconds(wall_time):
            raise ValueError("database lock wait cannot exceed the full gpu_worker Slurm wall time")
    preset_str = user.folding_release_preset or "public"
    try:
        release_preset = FoldingReleasePreset(preset_str)
    except ValueError as exc:
        raise ValueError(f"unsupported folding_release_preset: {preset_str!r}") from exc
    return ResolvedClusterProfile(
        name=cluster_name,
        owner=user.owner,
        project_root=user.project_root,
        output_root=user.output_root,
        staging_root=user.staging_root,
        afdb_toolkit_repo=user.afdb_toolkit_repo,
        orchestration_repo=user.orchestration_repo,
        image=user.image,
        transport=user.transport,
        ssh_target=user.ssh_target,
        probe_root=user.probe_root,
        source_bundle_root=user.source_bundle_root,
        runtime_image_cache_root=user.runtime_image_cache_root,
        runtime_image_policy=user.runtime_image_policy,
        runtime_qualification_root=user.runtime_qualification_root,
        runtime_qualification_control_root=control_root,
        governed_package_root=(
            user.governed_package_root
            or (str(Path(user.runtime_qualification_root) / "packages") if user.runtime_qualification_root else None)
        ),
        runtime_qualification_expires_hours=user.runtime_qualification_expires_hours,
        preprocessing_runtime=user.preprocessing_runtime,
        postprocessing_credential_mounts=user.postprocessing_credential_mounts,
        database_sets=user.database_sets,
        database_access_policies=user.database_access_policies,
        database_staging=staging,
        database_acceptance_cache=acceptance_cache,
        extra_mounts=user.extra_mounts,
        folding_release_preset=release_preset,
        folding_backend_images={entry.backend: entry.image for entry in user.folding_backend_images},
        folding_backend_assets={entry.backend: entry.to_snapshot() for entry in user.folding_backend_assets},
        mount_orchestration_source=user.mount_orchestration_source,
        account=account,
        resources=resources,
    )


def resolve_database_manifest_path(profile: ResolvedClusterProfile, selection: DatabaseSetSelection) -> Path:
    """Resolve one allowed Plan selection to its exact absolute manifest path."""
    if selection.requested_policy not in profile.database_access_policies:
        raise ValueError(
            f"Cluster Profile {profile.name!r} does not permit database policy {selection.requested_policy.value!r}"
        )
    if (
        selection.requested_policy in {DatabaseAccessPolicy.STAGE_REQUIRED, DatabaseAccessPolicy.STAGE_PREFERRED}
        and profile.database_staging is None
    ):
        raise ValueError(
            f"Cluster Profile {profile.name!r} requires complete database staging authority "
            f"for policy {selection.requested_policy.value!r}"
        )
    matches = tuple(
        item
        for item in profile.database_sets
        if (item.identifier, item.version) == (selection.database_set.identifier, selection.database_set.version)
    )
    if len(matches) != 1:
        raise ValueError("Cluster Profile must map the selected Database Set identity exactly once")
    return Path(matches[0].manifest_path)


def validate_folding_backend_asset_mount_coverage(
    assets: FoldingBackendAssetsSnapshot,
    extra_mounts: tuple[ProfileMount, ...],
) -> None:
    """Require every selected asset path to be covered by an exact read_only mount target.

    Match is on exact ``target`` equality (never inferred from ``source``) and
    only ``read_only=True`` mounts count. "Covered" means at
    least one matching target; duplicate/conflicting targets are rejected at
    render time by e09s03.
    """
    for name in (
        "chain_manifest_csv",
        "openfold_model_dir",
        "colabfold_weights_dir",
        "bioir_checkpoint",
        "bioir_monomer_checkpoint",
    ):
        path = getattr(assets, name)
        if path is None:
            continue
        if not any(mount.target == path and mount.read_only for mount in extra_mounts):
            raise ValueError(
                f"folding backend asset {name} path {path!r} must be covered by a read_only extra_mounts target"
            )


def _database_staging_snapshot(user: UserClusterProfile) -> DatabaseProfileStagingSnapshot | None:
    values = (
        user.database_cache_root,
        user.database_cache_unix_user,
        user.database_cache_filesystem_type,
        user.database_cache_reserve_bytes,
        user.database_lock_wait_seconds,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("database staging snapshot requires all cache fields")
    assert user.database_cache_root is not None
    assert user.database_cache_unix_user is not None
    assert user.database_cache_filesystem_type is not None
    assert user.database_cache_reserve_bytes is not None
    assert user.database_lock_wait_seconds is not None
    return DatabaseProfileStagingSnapshot(
        cache_root=user.database_cache_root,
        unix_user=user.database_cache_unix_user,
        expected_filesystem_type=user.database_cache_filesystem_type,
        reserve_bytes=user.database_cache_reserve_bytes,
        lock_wait_seconds=user.database_lock_wait_seconds,
    )


def _load_clusters(path: Path | None, *, document: bytes | None = None) -> Mapping[str, object]:
    if path is None:
        if document is not None:
            raise ValueError("bundled Cluster Profile templates do not accept an external document")
        data = yaml.safe_load(_TEMPLATE_RESOURCE.read_bytes())
        source = str(_TEMPLATE_RESOURCE)
    else:
        data = yaml.safe_load(path.read_bytes() if document is None else document)
        source = str(path)
    if not isinstance(data, Mapping):
        msg = f"Expected Cluster Profile YAML mapping in {source}"
        raise TypeError(msg)
    clusters = data.get("clusters")
    if not isinstance(clusters, Mapping):
        msg = f"Expected top-level clusters mapping in {source}"
        raise TypeError(msg)
    return clusters


def _generic_template_shape(clusters: Mapping[str, object]) -> object | None:
    """Return the bundled synthetic ``example`` shape, when present.

    The generic shape is only ever used for explicit synthetic example names
    (``example`` / ``example-*``). Arbitrary user cluster names must fully
    author their own account and resources in the user Cluster Profile.
    """
    return clusters.get("example")


def _is_example_name(cluster_name: str) -> bool:
    """Return True when *cluster_name* is an explicit synthetic example name."""
    return cluster_name == "example" or cluster_name.startswith("example-")


def _selected_cluster_mapping(clusters: Mapping[str, object], cluster_name: str) -> Mapping[str, object]:
    value = clusters[cluster_name]
    if not isinstance(value, Mapping):
        msg = f"Expected cluster profile {cluster_name!r} to be a mapping"
        raise TypeError(msg)
    return value


def _merge_resources(
    template_resources: Mapping[str, SlurmResources],
    user_resources: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, SlurmResources]:
    merged: dict[str, SlurmResources] = dict(template_resources)
    for name, override in user_resources.items():
        base = merged[name].model_dump(mode="json", exclude_none=True) if name in merged else {}
        base.update(dict(override))
        try:
            merged[name] = SlurmResources.model_validate(base)
        except ValidationError as exc:
            msg = f"Invalid resource override for Cluster Profile resource {name!r}: {exc}"
            raise ValueError(msg) from exc
    for name, resource in merged.items():
        _reject_conflicting_legacy_topology(resource, name)
    return merged


def _reject_conflicting_legacy_topology(resource: SlurmResources, name: str) -> None:
    if resource.nodes is None:
        return
    if resource.array is not None:
        raise ValueError(
            f"Cluster Profile resource {name!r} authors typed topology (nodes) together with a legacy array string"
        )
    if resource.gres is not None:
        raise ValueError(
            f"Cluster Profile resource {name!r} authors typed topology (nodes) together with a legacy gres string"
        )


__all__ = [
    "ClusterProfilePaths",
    "DatabaseSetManifestProfile",
    "FoldingBackendAssetsProfile",
    "PostprocessingCredentialMountProfile",
    "PreprocessingRuntimeProfile",
    "ProfileMount",
    "ResolvedClusterProfile",
    "TransportKind",
    "resolve_cluster_profile",
    "resolve_database_manifest_path",
    "validate_folding_backend_asset_mount_coverage",
]
