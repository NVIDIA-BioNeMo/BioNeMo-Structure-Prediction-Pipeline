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

"""Tests for the BioIR release preset and per-backend image override."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bspp.orchestration.contract.folding_execution import (
    OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR,
    FoldingBackendAssetsSnapshot,
)
from bspp.orchestration.contract.folding_release import (
    FoldingBackendImageOverride,
    FoldingReleasePreset,
    resolve_folding_release_preset,
)
from bspp.orchestration.contract.phase import (
    FOLDING_BACKENDS,
    FoldingResolvedClusterSnapshot,
    PhaseSlurmResources,
    canonical_mapping_digest,
    folding_phase_runspec_from_mapping,
    folding_resolved_cluster_snapshot_from_mapping,
)
from bspp.orchestration.control.folding_phase_adapter import _folding_cluster_snapshot
from bspp.orchestration.control.folding_phase_types import (
    FoldingBackendImageSelection,
    FoldingPhaseAttemptOperationalSelection,
)
from bspp.orchestration.control.phase_materialization import _resolve_folding_operational_selection
from bspp.orchestration.control.profiles import resolve_cluster_profile


def _default_preset() -> FoldingReleasePreset:
    """The tree's current default preset (the first enum member)."""
    return next(iter(FoldingReleasePreset))


def _default_preset_str() -> str:
    return _default_preset().value


def _has_internal() -> bool:
    """True when the INTERNAL preset still exists (internal main tree)."""
    return hasattr(FoldingReleasePreset, "INTERNAL")


def _write_template(tmp_path: Path, *, resources: dict | None = None) -> Path:
    """Write a minimal cluster profile template YAML with the given resources."""
    data = {
        "clusters": {
            "example-cluster": {
                "account": "bspp",
                "resources": resources
                or {
                    "control_cpu": {
                        "partition": "cpu",
                        "cpus_per_task": 4,
                        "memory": "16G",
                        "time": "01:00:00",
                    },
                    "gpu_worker": {
                        "partition": "gpu",
                        "cpus_per_task": 8,
                        "memory": "64G",
                        "time": "04:00:00",
                        "gres": "gpu:1",
                    },
                },
            }
        }
    }
    path = tmp_path / "templates.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _write_profile(
    tmp_path: Path,
    *,
    folding_release_preset: str | None = None,
    folding_backend_images: list[dict] | None = None,
    folding_backend_assets: list[dict] | None = None,
    extra_mounts: list[dict] | None = None,
) -> Path:
    """Write a minimal user cluster profile YAML."""
    profile: dict = {
        "owner": "bspp",
        "project_root": "/cluster/bspp",
        "output_root": "/cluster/bspp/output",
        "staging_root": "/cluster/bspp/staging",
        "orchestration_repo": "/cluster/bspp/orchestration",
        "image": "registry.example.com/bspp:latest",
        "transport": "ssh",
        "ssh_target": "example-cluster-oci-dc-02.example-cluster-oci-iad.nvidia.com",
        "account": "bspp",
    }
    if folding_release_preset is not None:
        profile["folding_release_preset"] = folding_release_preset
    if folding_backend_images is not None:
        profile["folding_backend_images"] = folding_backend_images
    if folding_backend_assets is not None:
        profile["folding_backend_assets"] = folding_backend_assets
    if extra_mounts is not None:
        profile["extra_mounts"] = extra_mounts
    data = {"clusters": {"example-cluster": profile}}
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _openfold_assets() -> list[dict]:
    return [
        {
            "backend": "openfold-cli",
            "chain_manifest_csv": "/assets/chains.csv",
            "openfold_model_dir": "/assets/models",
        }
    ]


def _openfold_mounts() -> list[dict]:
    return [
        {"source": "/assets/chains.csv", "target": "/assets/chains.csv", "read_only": True},
        {"source": "/assets/models", "target": "/assets/models", "read_only": True},
    ]


# ---------------------------------------------------------------------------
# Test 1: Pure resolver — INTERNAL
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_internal(), reason="INTERNAL preset is stripped in the sanitized public tree")
def test_resolve_internal_preset():
    r = resolve_folding_release_preset(FoldingReleasePreset.INTERNAL)
    assert r.preset == FoldingReleasePreset.INTERNAL
    assert r.package_source == "pypi"
    assert r.package_spec == "bionemo-ir==0.1.0"
    assert r.base_image == "nvidia/cuda:13.0.3-devel-ubuntu24.04"
    assert r.checkpoint_source == "fetch-weights-script"
    assert r.checkpoint_fetch_command == "scripts/fetch_weights.sh --model <name>"
    assert r.public_repo_url == "https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime"
    assert r.docs_url == "https://docs.nvidia.com/bionemo/inference-runtime/overview/"


# ---------------------------------------------------------------------------
# Test 2: Pure resolver — PUBLIC
# ---------------------------------------------------------------------------


def test_resolve_public_preset():
    r = resolve_folding_release_preset(FoldingReleasePreset.PUBLIC)
    assert r.preset == FoldingReleasePreset.PUBLIC
    assert r.package_source == "pypi"
    assert r.package_spec == "bionemo-ir==0.1.0"
    assert r.base_image == "nvidia/cuda:13.0.3-devel-ubuntu24.04"
    assert r.checkpoint_source == "fetch-weights-script"
    assert r.checkpoint_fetch_command == "scripts/fetch_weights.sh --model <name>"
    assert r.public_repo_url == "https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime"
    assert r.docs_url == "https://docs.nvidia.com/bionemo/inference-runtime/overview/"


# ---------------------------------------------------------------------------
# Test 3: Enum rejects unknown preset
# ---------------------------------------------------------------------------


def test_enum_rejects_unknown_preset():
    with pytest.raises(ValueError, match="is not a valid FoldingReleasePreset"):
        FoldingReleasePreset("unknown")


# ---------------------------------------------------------------------------
# Test 4: FoldingBackendImageOverride validation
# ---------------------------------------------------------------------------


def test_backend_image_override_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        FoldingBackendImageOverride(backend="nonexistent", image="/path/to/img.sqsh")


def test_backend_image_override_rejects_empty_image():
    with pytest.raises(ValidationError):
        FoldingBackendImageOverride(backend="bioir", image="")


# ---------------------------------------------------------------------------
# Test 5: UserClusterProfile duplicate-backend rejection
# ---------------------------------------------------------------------------


def test_user_profile_rejects_duplicate_backends(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(
        tmp_path,
        folding_backend_images=[
            {"backend": "bioir", "image": "/a.sqsh"},
            {"backend": "bioir", "image": "/b.sqsh"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate backends"):
        resolve_cluster_profile("example-cluster", config_path=config, template_path=template)


# ---------------------------------------------------------------------------
# Test 6: resolve_cluster_profile preset defaulting and validation
# ---------------------------------------------------------------------------


def test_profile_defaults_to_internal_preset(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(tmp_path)
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    assert profile.folding_release_preset == _default_preset()


def test_profile_resolves_public_preset(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(tmp_path, folding_release_preset="public")
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    assert profile.folding_release_preset == FoldingReleasePreset.PUBLIC


def test_profile_rejects_unknown_preset(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(tmp_path, folding_release_preset="unknown")
    with pytest.raises(ValueError, match="unsupported folding_release_preset"):
        resolve_cluster_profile("example-cluster", config_path=config, template_path=template)


# ---------------------------------------------------------------------------
# Test 7: Per-backend image override in resolved profile
# ---------------------------------------------------------------------------


def test_profile_resolves_backend_image_override(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(
        tmp_path,
        folding_backend_images=[
            {"backend": "bioir", "image": "/custom/bioir.sqsh"},
        ],
    )
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    assert profile.folding_backend_images["bioir"] == "/custom/bioir.sqsh"
    assert "openfold-cli" not in profile.folding_backend_images


# ---------------------------------------------------------------------------
# Test 8: _resolve_folding_operational_selection with override
# ---------------------------------------------------------------------------


def test_operational_selection_uses_backend_override(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(
        tmp_path,
        folding_backend_images=[
            {"backend": "bioir", "image": "/custom/bioir.sqsh"},
        ],
        folding_backend_assets=_openfold_assets(),
        extra_mounts=_openfold_mounts(),
    )
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    operational = _resolve_folding_operational_selection(profile, "openfold-cli")
    assert operational.backend_images.image_for_backend("bioir") == "/custom/bioir.sqsh"
    with pytest.raises(ValueError, match="no kernel image selected"):
        operational.backend_images.image_for_backend("openfold-cli")
    assert operational.release_preset == _default_preset()


# ---------------------------------------------------------------------------
# Test 9: Default behavior — no override
# ---------------------------------------------------------------------------


def test_operational_selection_requires_backend_override(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(tmp_path, folding_backend_assets=_openfold_assets(), extra_mounts=_openfold_mounts())
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    with pytest.raises(ValueError, match="non-empty"):
        _resolve_folding_operational_selection(profile, "openfold-cli")


# ---------------------------------------------------------------------------
# Test 10: Observable downstream effect — preset changes cluster snapshot digest
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_internal(), reason="requires both INTERNAL and PUBLIC presets")
def test_preset_changes_cluster_snapshot_digest():
    resources = PhaseSlurmResources(partition="cpu", cpus_per_task=4, memory="16G", time="01:00:00")
    fold_resources = PhaseSlurmResources(partition="gpu", cpus_per_task=8, memory="64G", time="04:00:00", gres="gpu:1")
    backend_images = FoldingBackendImageSelection(backend_images={b: "registry/img:latest" for b in FOLDING_BACKENDS})
    base_kwargs = dict(
        profile_name="folding-gpu",
        owner="bspp",
        transport="ssh",
        ssh_target="example-cluster",
        account="bspp",
        project_root="/p",
        staging_root="/s",
        orchestration_repo="/o",
        runtime_image="registry/bspp:latest",
        backend_images=backend_images,
        resources=resources,
        fold_resources=fold_resources,
    )
    op_internal = FoldingPhaseAttemptOperationalSelection(release_preset=FoldingReleasePreset.INTERNAL, **base_kwargs)
    op_public = FoldingPhaseAttemptOperationalSelection(release_preset=FoldingReleasePreset.PUBLIC, **base_kwargs)
    snap_internal = _folding_cluster_snapshot(op_internal)
    snap_public = _folding_cluster_snapshot(op_public)
    assert snap_internal.release_preset == "internal"
    assert snap_public.release_preset == "public"
    digest_internal = canonical_mapping_digest(snap_internal.to_mapping())
    digest_public = canonical_mapping_digest(snap_public.to_mapping())
    assert digest_internal != digest_public


# ---------------------------------------------------------------------------
# Test 11: FoldingResolvedClusterSnapshot round-trip with release_preset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_internal(), reason="requires a non-default preset to test emission")
def test_cluster_snapshot_roundtrip_with_preset():
    cluster = FoldingResolvedClusterSnapshot(
        profile_name="p",
        owner="o",
        transport="ssh",
        ssh_target="example-cluster",
        account="a",
        project_root="/p",
        staging_root="/s",
        orchestration_repo="/o",
        runtime_image="img",
        extra_mounts=(),
        release_preset="public",
    )
    assert folding_resolved_cluster_snapshot_from_mapping(cluster.to_mapping()) == cluster
    assert cluster.to_mapping()["release_preset"] == "public"


# ---------------------------------------------------------------------------
# Test 12: FoldingResolvedClusterSnapshot defaults to "internal" when absent
# ---------------------------------------------------------------------------


def test_cluster_snapshot_defaults_to_internal():
    mapping = {
        "schema_version": 1,
        "profile_name": "p",
        "owner": "o",
        "transport": "ssh",
        "ssh_target": "example-cluster",
        "account": "a",
        "project_root": "/p",
        "staging_root": "/s",
        "orchestration_repo": "/o",
        "runtime_image": "img",
        "extra_mounts": [],
    }
    cluster = folding_resolved_cluster_snapshot_from_mapping(mapping)
    assert cluster.release_preset == _default_preset_str()


def test_internal_snapshot_omits_release_preset_preserving_legacy_digest():
    """The default preset must not be emitted, so pre-field RunSpecs
    re-serialize to their original bytes and keep their canonical digests."""
    mapping = {
        "schema_version": 1,
        "profile_name": "p",
        "owner": "o",
        "transport": "ssh",
        "ssh_target": "example-cluster",
        "account": "a",
        "project_root": "/p",
        "staging_root": "/s",
        "orchestration_repo": "/o",
        "runtime_image": "img",
        "extra_mounts": [],
    }
    cluster = folding_resolved_cluster_snapshot_from_mapping(mapping)
    assert cluster.release_preset == _default_preset_str()
    assert "release_preset" not in cluster.to_mapping()


# ---------------------------------------------------------------------------
# Test 13: validate_folding_plan_runspec_binding preserves release_preset
# ---------------------------------------------------------------------------


def test_binding_preserves_release_preset(tmp_path: Path):
    """A materialized RunSpec with release_preset='public' must satisfy binding validation."""
    # Build a profile with public preset
    template = _write_template(tmp_path)
    config = _write_profile(
        tmp_path,
        folding_release_preset="public",
        folding_backend_images=[{"backend": "openfold-cli", "image": "/custom/openfold-cli.sqsh"}],
        folding_backend_assets=_openfold_assets(),
        extra_mounts=_openfold_mounts(),
    )
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    operational = _resolve_folding_operational_selection(profile, "openfold-cli")
    assert operational.release_preset == FoldingReleasePreset.PUBLIC

    # Materialize a RunSpec using the operational selection
    from bspp.orchestration.contract.folding_input import MsaSetConsumption
    from bspp.orchestration.contract.preprocessing_handoff import (
        BundledMemberVerification,
        MsaArtifactSetManifest,
        MsaChunkManifestReference,
        VerifiedLocalBundledArtifactLocation,
        msa_artifact_set_id,
        verified_local_bundled_artifact_location_id,
    )
    from bspp.orchestration.control.folding_phase_adapter import (
        materialize_folding_attempt_runspec,
        validate_folding_plan_runspec_binding,
    )

    chunk = MsaChunkManifestReference(
        chunk_name="foo_tranche00_00001.fa",
        logical_path="chunks/foo_tranche00_00001.json",
        sha256="f" * 64,
        member_count=1,
        logical_bytes=1,
    )
    manifest = MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((chunk,), 1, 1, member_lengths=(1,)),
        chunks=(chunk,),
        member_count=1,
        logical_bytes=1,
        member_lengths=(1,),
    )
    artifact_set_id = manifest.artifact_set_id
    member_name = "AFDB_AF-0000000000000001.a3m"
    member_path = f"a3ms/{member_name}"
    msa_set = MsaSetConsumption(
        artifact_set_id=artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(member_path,),
        requires_paired_query_header=True,
    )

    # Create a local bundled artifact location
    bundle_path = tmp_path / "msa-set" / "msa-set.tar.lz4"
    tar_path = tmp_path / "msa-set" / "msa-set.tar"
    bundle_path.parent.mkdir(parents=True)
    bundle_bytes = b"bspp-fixture-lz4-bundle\n"
    tar_bytes = b"bspp-fixture-tar\n"
    bundle_path.write_bytes(bundle_bytes)
    tar_path.write_bytes(tar_bytes)
    import hashlib

    lz4_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    tar_sha256 = hashlib.sha256(tar_bytes).hexdigest()
    members = (
        BundledMemberVerification(
            logical_path=member_path,
            member_name=member_name,
            raw_member_name=member_name,
            size_bytes=1,
            sha256="b" * 64,
        ),
    )
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=Path(bundle_path).as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=tar_sha256,
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=lz4_sha256,
        raw_tar_members=(member_name,),
        members=members,
    )
    location = VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=Path(bundle_path).as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=tar_sha256,
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=lz4_sha256,
        raw_tar_members=(member_name,),
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )

    from bspp.orchestration.contract.phase import FoldingPhasePlan, FoldingPhasePlanPayload

    phase_plan = FoldingPhasePlan(
        target_cluster="example-cluster",
        input_location=location,
        payload=FoldingPhasePlanPayload(msa_set=msa_set, backend="openfold-cli", msa_set_manifest=manifest),
    )

    runspec = materialize_folding_attempt_runspec(
        phase_run_id="phase-run-" + "a" * 32,
        attempt_id="attempt-0001",
        phase_plan=phase_plan,
        materialized_at="2026-01-01T00:00:00Z",
        operational=operational,
    )

    # The RunSpec's cluster snapshot should carry release_preset="public"
    assert runspec.cluster.release_preset == "public"

    # validate_folding_plan_runspec_binding must not raise (no TypeError from missing arg)
    validate_folding_plan_runspec_binding(phase_plan, runspec)


# ---------------------------------------------------------------------------
# Test 14: FoldingResolvedClusterSnapshot rejects invalid release_preset
# ---------------------------------------------------------------------------


def test_cluster_snapshot_rejects_invalid_preset():
    with pytest.raises(ValueError, match="unsupported folding release_preset"):
        FoldingResolvedClusterSnapshot(
            profile_name="p",
            owner="o",
            transport="ssh",
            ssh_target="example-cluster",
            account="a",
            project_root="/p",
            staging_root="/s",
            orchestration_repo="/o",
            runtime_image="img",
            extra_mounts=(),
            release_preset="bogus",
        )


def test_cluster_snapshot_from_mapping_rejects_invalid_preset():
    mapping = {
        "schema_version": 1,
        "profile_name": "p",
        "owner": "o",
        "transport": "ssh",
        "ssh_target": "example-cluster",
        "account": "a",
        "project_root": "/p",
        "staging_root": "/s",
        "orchestration_repo": "/o",
        "runtime_image": "img",
        "extra_mounts": [],
        "release_preset": "bogus",
    }
    with pytest.raises(ValueError, match="unsupported folding release_preset"):
        folding_resolved_cluster_snapshot_from_mapping(mapping)


# ---------------------------------------------------------------------------
# Test 15: folding_phase_runspec_from_mapping loads a pre-field mapping
# ---------------------------------------------------------------------------


def test_folding_phase_runspec_from_mapping_pre_field_defaults_internal():
    """A RunSpec mapping without release_preset in cluster loads and defaults to internal."""
    # Reuse the contract test fixture pattern to build a complete mapping
    from tests.test_phase_folding_contract import make_plan, make_runspec

    plan = make_plan()
    runspec = make_runspec(plan)
    mapping = runspec.to_mapping()

    # Remove release_preset from the cluster sub-mapping to simulate pre-field authority
    cluster_mapping = dict(mapping["cluster"])
    cluster_mapping.pop("release_preset", None)
    mapping["cluster"] = cluster_mapping

    loaded = folding_phase_runspec_from_mapping(mapping)
    assert loaded.cluster.release_preset == _default_preset_str()
    assert loaded == runspec


# ---------------------------------------------------------------------------
# Backend assets
# ---------------------------------------------------------------------------


def test_profile_resolves_backend_tagged_assets_mapping(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(tmp_path, folding_backend_assets=_openfold_assets(), extra_mounts=_openfold_mounts())
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    snapshot = profile.folding_backend_assets["openfold-cli"]
    assert isinstance(snapshot, FoldingBackendAssetsSnapshot)
    assert snapshot.chain_manifest_csv == "/assets/chains.csv"
    assert snapshot.openfold_model_dir == "/assets/models"


def test_profile_rejects_duplicate_backend_assets(tmp_path: Path):
    template = _write_template(tmp_path)
    assets = [
        *_openfold_assets(),
        {
            "backend": "openfold-cli",
            "chain_manifest_csv": "/assets/other.csv",
            "openfold_model_dir": "/assets/other-models",
        },
    ]
    config = _write_profile(tmp_path, folding_backend_assets=assets, extra_mounts=_openfold_mounts())
    with pytest.raises(ValueError, match="duplicate backends"):
        resolve_cluster_profile("example-cluster", config_path=config, template_path=template)


def test_profile_rejects_unknown_asset_key(tmp_path: Path):
    template = _write_template(tmp_path)
    assets = [
        {
            "backend": "openfold-cli",
            "chain_manifest_csv": "/assets/chains.csv",
            "openfold_model_dir": "/assets/models",
            "model_fn_factory": "foo",
        }
    ]
    config = _write_profile(tmp_path, folding_backend_assets=assets, extra_mounts=_openfold_mounts())
    with pytest.raises(ValueError, match="model_fn_factory"):
        resolve_cluster_profile("example-cluster", config_path=config, template_path=template)


def test_profile_rejects_relative_asset_path(tmp_path: Path):
    template = _write_template(tmp_path)
    assets = [
        {
            "backend": "openfold-cli",
            "chain_manifest_csv": "relative/chains.csv",
            "openfold_model_dir": "/assets/models",
        }
    ]
    config = _write_profile(tmp_path, folding_backend_assets=assets, extra_mounts=_openfold_mounts())
    with pytest.raises(ValueError, match="absolute"):
        resolve_cluster_profile("example-cluster", config_path=config, template_path=template)


def test_profile_rejects_trt_factory_field(tmp_path: Path):
    template = _write_template(tmp_path)
    assets = [
        {
            "backend": "openfold-trt",
            "chain_manifest_csv": "/assets/chains.csv",
            "trt_model_fn": "foo",
        }
    ]
    config = _write_profile(tmp_path, folding_backend_assets=assets, extra_mounts=_openfold_mounts())
    with pytest.raises(ValueError, match="trt_model_fn"):
        resolve_cluster_profile("example-cluster", config_path=config, template_path=template)


def test_asset_mount_coverage_requires_read_only_target(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(
        tmp_path,
        folding_backend_images=[{"backend": "openfold-cli", "image": "/custom/openfold-cli.sqsh"}],
        folding_backend_assets=_openfold_assets(),
        extra_mounts=[
            {"source": "/assets/chains.csv", "target": "/assets/chains.csv"},
            {"source": "/assets/models", "target": "/assets/models"},
        ],
    )
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    with pytest.raises(ValueError, match="read_only extra_mounts target"):
        _resolve_folding_operational_selection(profile, "openfold-cli")

    config = _write_profile(
        tmp_path,
        folding_backend_images=[{"backend": "openfold-cli", "image": "/custom/openfold-cli.sqsh"}],
        folding_backend_assets=_openfold_assets(),
        extra_mounts=_openfold_mounts(),
    )
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    operational = _resolve_folding_operational_selection(profile, "openfold-cli")
    assert operational.assets is not None


def test_resolve_operational_selection_fails_closed_on_missing_assets(tmp_path: Path):
    template = _write_template(tmp_path)
    config = _write_profile(
        tmp_path,
        folding_backend_images=[{"backend": "openfold-cli", "image": "/custom/openfold-cli.sqsh"}],
    )
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    with pytest.raises(ValueError, match="does not resolve folding backend assets"):
        _resolve_folding_operational_selection(profile, "openfold-cli")


def test_resolve_operational_selection_trt_deferred_model_fn(tmp_path: Path):
    """openfold-trt reaches the stable deferred-model_fn error even without a TRT asset stanza."""
    template = _write_template(tmp_path)
    config = _write_profile(tmp_path, folding_backend_assets=_openfold_assets(), extra_mounts=_openfold_mounts())
    profile = resolve_cluster_profile("example-cluster", config_path=config, template_path=template)
    with pytest.raises(ValueError, match=re.escape(OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR)):
        _resolve_folding_operational_selection(profile, "openfold-trt")


def test_asset_change_changes_cluster_snapshot_digest(tmp_path: Path):
    template = _write_template(tmp_path)
    config_a = _write_profile(
        tmp_path,
        folding_backend_images=[{"backend": "openfold-cli", "image": "/custom/openfold-cli.sqsh"}],
        folding_backend_assets=_openfold_assets(),
        extra_mounts=_openfold_mounts(),
    )
    profile_a = resolve_cluster_profile("example-cluster", config_path=config_a, template_path=template)
    op_a = _resolve_folding_operational_selection(profile_a, "openfold-cli")

    alternate = [
        {
            "backend": "openfold-cli",
            "chain_manifest_csv": "/assets/other.csv",
            "openfold_model_dir": "/assets/other-models",
        }
    ]
    alternate_mounts = [
        {"source": "/assets/other.csv", "target": "/assets/other.csv", "read_only": True},
        {"source": "/assets/other-models", "target": "/assets/other-models", "read_only": True},
    ]
    config_b = _write_profile(
        tmp_path,
        folding_backend_images=[{"backend": "openfold-cli", "image": "/custom/openfold-cli.sqsh"}],
        folding_backend_assets=alternate,
        extra_mounts=alternate_mounts,
    )
    profile_b = resolve_cluster_profile("example-cluster", config_path=config_b, template_path=template)
    op_b = _resolve_folding_operational_selection(profile_b, "openfold-cli")

    snap_a = _folding_cluster_snapshot(op_a)
    snap_b = _folding_cluster_snapshot(op_b)
    assert canonical_mapping_digest(snap_a.to_mapping()) != canonical_mapping_digest(snap_b.to_mapping())
