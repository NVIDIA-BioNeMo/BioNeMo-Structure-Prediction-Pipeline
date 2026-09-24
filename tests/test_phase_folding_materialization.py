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

"""Local-fixture tests for the folding Phase adapter.

No cluster, no subprocess, and no runtime imports except the deliberate Track C
``load_canonical_pair_index`` import used to prove the control-side index
builder emits byte/schema-identical JSON.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot
from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.folding_release import FoldingReleasePreset
from bspp.orchestration.contract.folding_shard import fold_shard_projection_document_bytes
from bspp.orchestration.contract.phase import (
    FoldingPhasePlan,
    FoldingPhasePlanPayload,
    FoldingPhaseRunSpec,
    PhaseMountSnapshot,
    PhaseSlurmResources,
    canonical_mapping_digest,
)
from bspp.orchestration.contract.phase_submission import (
    PhaseSubmissionIntendedPayload,
    phase_action_scheduler_correlation_token,
    phase_submission_intended_payload_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
    msa_artifact_set_id,
    msa_artifact_set_manifest_from_mapping,
    verified_local_bundled_artifact_location_id,
    verified_remote_bundled_artifact_location_id,
)
from bspp.orchestration.contract.runspec import SlurmResources
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.control.folding_phase_adapter import (
    materialize_folding_attempt_runspec,
    render_folding_submission_intent,
    validate_folding_action_evidence,
    validate_folding_plan_runspec_binding,
)
from bspp.orchestration.control.folding_phase_types import (
    FoldingBackendImageSelection,
    FoldingPhaseAttemptOperationalSelection,
)
from bspp.orchestration.control.folding_shard import (
    FOLD_SHARD_LPT_VERSION,
    derive_fold_shard_projection,
    fold_shard_projection_from_runspec,
)
from bspp.orchestration.control.phase_authority import PhaseAuthorityStore

ARTIFACT_SET_ID = "sha256:" + "a" * 64
MEMBER_NAME = "AFDB_AF-0000000000000001.a3m"
MEMBER_PATH = f"a3ms/{MEMBER_NAME}"
PHASE_RUN_ID = "phase-run-" + "a" * 32
ATTEMPT_ID = "attempt-0001"
RUNTIME_IMAGE = "registry/bspp-runtime:latest"
OPENFOLD_CLI_IMAGE = "registry/openfold-cli:latest"
ACTION_IDS = [
    "msa-flatten-000001",
    "split-000001",
    "preprocess-000001",
    "fold-000001",
    "canonical-pair-000001",
]


def _default_preset() -> FoldingReleasePreset:
    """The tree's current default folding release preset (INTERNAL on main, PUBLIC after sanitization)."""
    return next(iter(FoldingReleasePreset))


def _bundled_location(artifact_set_id: str = ARTIFACT_SET_ID) -> VerifiedLocalBundledArtifactLocation:
    tar_path = "/bspp-fixture/msa-set/msa-set.tar"
    bundle_path = "/bspp-fixture/msa-set/msa-set.tar.lz4"
    bundle_uri = Path(bundle_path).as_uri()
    members = (
        BundledMemberVerification(
            logical_path=MEMBER_PATH,
            member_name=MEMBER_NAME,
            raw_member_name=MEMBER_NAME,
            size_bytes=1,
            sha256="b" * 64,
        ),
    )
    raw_tar_members = (MEMBER_NAME,)
    tar_size_bytes = 1
    tar_sha256 = "c" * 64
    lz4_size_bytes = 1
    lz4_sha256 = "d" * 64
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=artifact_set_id,
        tar_path=tar_path,
        bundle_path=bundle_path,
        bundle_uri=bundle_uri,
        tar_size_bytes=tar_size_bytes,
        tar_sha256=tar_sha256,
        lz4_size_bytes=lz4_size_bytes,
        lz4_sha256=lz4_sha256,
        raw_tar_members=raw_tar_members,
        members=members,
    )
    return VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set_id,
        tar_path=tar_path,
        bundle_path=bundle_path,
        bundle_uri=bundle_uri,
        tar_size_bytes=tar_size_bytes,
        tar_sha256=tar_sha256,
        lz4_size_bytes=lz4_size_bytes,
        lz4_sha256=lz4_sha256,
        raw_tar_members=raw_tar_members,
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )


def _remote_location(artifact_set_id: str = ARTIFACT_SET_ID) -> VerifiedRemoteBundledArtifactLocation:
    bundle_uri = "s3://example-bucket-fixture/msa-set/msa-set.tar.lz4"
    members = (
        BundledMemberVerification(
            logical_path=MEMBER_PATH,
            member_name=MEMBER_NAME,
            raw_member_name=MEMBER_NAME,
            size_bytes=1,
            sha256="b" * 64,
        ),
    )
    raw_tar_members = (MEMBER_NAME,)
    tar_size_bytes = 1
    tar_sha256 = "c" * 64
    lz4_size_bytes = 1
    lz4_sha256 = "d" * 64
    location_id = verified_remote_bundled_artifact_location_id(
        artifact_set_id=artifact_set_id,
        bundle_uri=bundle_uri,
        tar_size_bytes=tar_size_bytes,
        tar_sha256=tar_sha256,
        lz4_size_bytes=lz4_size_bytes,
        lz4_sha256=lz4_sha256,
        raw_tar_members=raw_tar_members,
        members=members,
    )
    return VerifiedRemoteBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set_id,
        bundle_uri=bundle_uri,
        tar_size_bytes=tar_size_bytes,
        tar_sha256=tar_sha256,
        lz4_size_bytes=lz4_size_bytes,
        lz4_sha256=lz4_sha256,
        raw_tar_members=raw_tar_members,
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )


def _manifest() -> MsaArtifactSetManifest:
    chunk = MsaChunkManifestReference(
        chunk_name="foo_tranche00_00001.fa",
        logical_path="chunks/foo_tranche00_00001.json",
        sha256="f" * 64,
        member_count=1,
        logical_bytes=1,
    )
    return MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((chunk,), 1, 1, member_lengths=(1,)),
        chunks=(chunk,),
        member_count=1,
        logical_bytes=1,
        member_lengths=(1,),
    )


def _legacy_manifest() -> MsaArtifactSetManifest:
    chunk = MsaChunkManifestReference(
        chunk_name="foo_tranche00_00001.fa",
        logical_path="chunks/foo_tranche00_00001.json",
        sha256="f" * 64,
        member_count=1,
        logical_bytes=1,
    )
    return MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((chunk,), 1, 1),
        chunks=(chunk,),
        member_count=1,
        logical_bytes=1,
    )


def _phase_plan(backend: str = "openfold-cli", *, with_manifest: bool = True) -> FoldingPhasePlan:
    manifest = _manifest() if with_manifest else None
    artifact_set_id = manifest.artifact_set_id if manifest is not None else ARTIFACT_SET_ID
    msa_set = MsaSetConsumption(
        artifact_set_id=artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(MEMBER_PATH,),
        requires_paired_query_header=True,
    )
    return FoldingPhasePlan(
        target_cluster="example-cluster",
        input_location=_bundled_location(artifact_set_id),
        payload=FoldingPhasePlanPayload(msa_set=msa_set, backend=backend, msa_set_manifest=manifest),
    )


def _assets(backend: str = "openfold-cli") -> FoldingBackendAssetsSnapshot:
    if backend == "openfold-cli":
        return FoldingBackendAssetsSnapshot(
            backend="openfold-cli",
            chain_manifest_csv="/assets/chains.csv",
            openfold_model_dir="/assets/models",
        )
    if backend == "openfold-trt":
        return FoldingBackendAssetsSnapshot(backend="openfold-trt", chain_manifest_csv="/assets/chains.csv")
    if backend == "bioir":
        return FoldingBackendAssetsSnapshot(
            backend="bioir",
            chain_manifest_csv="/assets/chains.csv",
            bioir_checkpoint="/assets/bioir.pt",
        )
    if backend == "colabfold":
        return FoldingBackendAssetsSnapshot(
            backend="colabfold",
            chain_manifest_csv="/assets/chains.csv",
            colabfold_weights_dir="/assets/colabfold",
        )
    raise ValueError(f"unsupported backend: {backend}")


def _operational(
    backend: str = "openfold-cli",
    extra_mounts: tuple[PhaseMountSnapshot, ...] = (),
) -> FoldingPhaseAttemptOperationalSelection:
    resources = PhaseSlurmResources(partition="cpu", cpus_per_task=4, memory="16G", time="01:00:00")
    fold_resources = PhaseSlurmResources(partition="gpu", cpus_per_task=8, memory="64G", time="04:00:00", gres="gpu:1")
    backend_images = FoldingBackendImageSelection(
        backend_images={
            "openfold-cli": OPENFOLD_CLI_IMAGE,
            "bioir": "registry/bioir:latest",
            "colabfold": "registry/colabfold:latest",
            "openfold-trt": "registry/openfold-trt:latest",
        }
    )
    return FoldingPhaseAttemptOperationalSelection(
        profile_name="folding-gpu",
        owner="bspp",
        transport="ssh",
        ssh_target="example-cluster",
        account="bspp",
        project_root="/lustre/bspp",
        staging_root="/lustre/bspp/staging",
        orchestration_repo="/lustre/bspp/orchestration",
        runtime_image=RUNTIME_IMAGE,
        backend_images=backend_images,
        release_preset=_default_preset(),
        resources=resources,
        fold_resources=fold_resources,
        assets=_assets(backend),
        extra_mounts=extra_mounts,
    )


def _materialized(backend: str = "openfold-cli"):
    return materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=_phase_plan(backend),
        materialized_at="2026-01-01T00:00:00Z",
        operational=_operational(backend),
    )


def _materialized_with(
    backend: str = "openfold-cli",
    *,
    extra_mounts: tuple[PhaseMountSnapshot, ...] = (),
    remote: bool = False,
):
    plan = _phase_plan(backend)
    if remote:
        plan = replace(plan, input_location=_remote_location(plan.payload.msa_set.artifact_set_id))
    return materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=plan,
        materialized_at="2026-01-01T00:00:00Z",
        operational=_operational(backend, extra_mounts=extra_mounts),
    )


def _prediction_pair_mapping(model_entity_id: str = "AF-0000000000000001") -> dict[str, object]:
    return {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "model_entity_id": model_entity_id,
        "tool_used": "OpenFold / AlphaFold-Multimer",
        "structure_path": f"{model_entity_id}-model_v1.pdb",
        "scores_path": f"{model_entity_id}-meta_v1.json",
        "scores": {
            "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
            "plddt": [0.5, 0.6],
            "pae": [[0.0, 0.1], [0.1, 0.0]],
            "max_pae": 0.1,
            "ptm": 0.9,
            "iptm": 0.8,
        },
    }


def _full_evidence() -> dict[str, dict[str, object]]:
    pair = _prediction_pair_mapping()
    return {
        "msa-flatten-000001": {"a3m_paths": [MEMBER_PATH]},
        "split-000001": {"chain_files": ["chain_1.a3m", "chain_2.a3m"]},
        "preprocess-000001": {"fasta_dir": "fasta", "alignment_dir": "alignments", "layout": "openfold"},
        "fold-000001": {"pairs": [pair]},
        "canonical-pair-000001": {
            "entries": [
                {"target_id": "target-1", "sequence_sha256": "f" * 64, "pair": pair},
            ]
        },
    }


def test_materialize_builds_exact_five_action_graph_and_linear_chain() -> None:
    runspec = _materialized()
    assert [action.action_id for action in runspec.payload.actions] == ACTION_IDS
    assert [action.action_kind for action in runspec.payload.actions] == [
        "msa-flatten",
        "split",
        "preprocess",
        "fold",
        "canonical-pair",
    ]
    for index, action in enumerate(runspec.payload.actions):
        if index == 0:
            assert action.dependencies == ()
        else:
            assert action.dependencies == (runspec.payload.actions[index - 1].action_id,)
    fold_action = runspec.payload.actions[3]
    assert fold_action.resources.gres == "gpu:1"
    assert runspec.payload.actions[0].resources.gres is None
    assert runspec.cluster.runtime_image == RUNTIME_IMAGE


def test_materialize_embeds_backend_and_kernel_image_on_fold_action() -> None:
    runspec = _materialized()
    fold_action = runspec.payload.actions[3]
    params = dict(fold_action.payload.params)
    assert params["backend"] == "openfold-cli"
    assert params["kernel_image"] == OPENFOLD_CLI_IMAGE


def test_materialize_stores_backend_assets_in_cluster_snapshot() -> None:
    runspec = _materialized()
    assert runspec.cluster.backend_assets == _assets("openfold-cli")


def test_materialize_asset_change_changes_cluster_digest_and_qualification_id() -> None:
    runspec_a = _materialized()
    alternate = FoldingBackendAssetsSnapshot(
        backend="openfold-cli",
        chain_manifest_csv="/assets/other-chains.csv",
        openfold_model_dir="/assets/other-models",
    )
    operational_b = replace(_operational(), assets=alternate)
    runspec_b = materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=_phase_plan(),
        materialized_at="2026-01-01T00:00:00Z",
        operational=operational_b,
    )
    assert canonical_mapping_digest(runspec_a.cluster.to_mapping()) != canonical_mapping_digest(
        runspec_b.cluster.to_mapping()
    )
    intent_a = render_folding_submission_intent(
        phase_runspec=runspec_a,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    intent_b = render_folding_submission_intent(
        phase_runspec=runspec_b,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    assert intent_a.qualification_tuple_id != intent_b.qualification_tuple_id


def test_materialize_passes_msa_set_manifest_into_runspec_payload() -> None:
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
    msa_set = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(MEMBER_PATH,),
        requires_paired_query_header=True,
    )
    plan = FoldingPhasePlan(
        target_cluster="example-cluster",
        input_location=_bundled_location(manifest.artifact_set_id),
        payload=FoldingPhasePlanPayload(msa_set=msa_set, backend="openfold-cli", msa_set_manifest=manifest),
    )
    runspec = materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=plan,
        materialized_at="2026-01-01T00:00:00Z",
        operational=_operational(),
    )
    assert runspec.payload.msa_set_manifest == manifest


def test_materialize_rejects_missing_root_manifest() -> None:
    """A manifest-less historical Plan is loadable but not executable."""
    with pytest.raises(ValueError, match="root MSA set manifest"):
        materialize_folding_attempt_runspec(
            phase_run_id=PHASE_RUN_ID,
            attempt_id=ATTEMPT_ID,
            phase_plan=_phase_plan(with_manifest=False),
            materialized_at="2026-01-01T00:00:00Z",
            operational=_operational(),
        )


def test_legacy_manifest_keeps_exact_id_and_deserializes_unchanged() -> None:
    legacy = _legacy_manifest()
    assert legacy.has_member_lengths() is False
    assert legacy.member_lengths is None
    assert msa_artifact_set_manifest_from_mapping(legacy.to_mapping()) == legacy
    chunk = legacy.chunks[0]
    assert legacy.artifact_set_id == msa_artifact_set_id((chunk,), 1, 1)


def test_enriched_manifest_gets_a_distinct_id() -> None:
    chunk = MsaChunkManifestReference(
        chunk_name="foo_tranche00_00001.fa",
        logical_path="chunks/foo_tranche00_00001.json",
        sha256="f" * 64,
        member_count=1,
        logical_bytes=1,
    )
    legacy_id = msa_artifact_set_id((chunk,), 1, 1)
    enriched_id = msa_artifact_set_id((chunk,), 1, 1, member_lengths=(1,))
    assert legacy_id != enriched_id


def test_materialize_rejects_legacy_msa_set_manifest() -> None:
    manifest = _legacy_manifest()
    msa_set = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(MEMBER_PATH,),
        requires_paired_query_header=True,
    )
    plan = FoldingPhasePlan(
        target_cluster="example-cluster",
        input_location=_bundled_location(manifest.artifact_set_id),
        payload=FoldingPhasePlanPayload(msa_set=msa_set, backend="openfold-cli", msa_set_manifest=manifest),
    )
    with pytest.raises(ValueError, match="legacy MSA manifest requires Runtime length enrichment"):
        materialize_folding_attempt_runspec(
            phase_run_id=PHASE_RUN_ID,
            attempt_id=ATTEMPT_ID,
            phase_plan=plan,
            materialized_at="2026-01-01T00:00:00Z",
            operational=_operational(),
        )


def test_materialize_accepts_enriched_msa_set_manifest() -> None:
    runspec = materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=_phase_plan(),
        materialized_at="2026-01-01T00:00:00Z",
        operational=_operational(),
    )
    assert runspec.payload.msa_set_manifest is not None
    assert runspec.payload.msa_set_manifest.has_member_lengths()


def test_member_lengths_count_violation_rejected() -> None:
    chunk = MsaChunkManifestReference(
        chunk_name="foo_tranche00_00001.fa",
        logical_path="chunks/foo_tranche00_00001.json",
        sha256="f" * 64,
        member_count=1,
        logical_bytes=1,
    )
    with pytest.raises(ValueError, match="member_lengths count must match member_count"):
        MsaArtifactSetManifest(
            artifact_set_id=msa_artifact_set_id((chunk,), 1, 1, member_lengths=(1, 2)),
            chunks=(chunk,),
            member_count=1,
            logical_bytes=1,
            member_lengths=(1, 2),
        )


def test_member_lengths_positivity_violation_rejected() -> None:
    chunk = MsaChunkManifestReference(
        chunk_name="foo_tranche00_00001.fa",
        logical_path="chunks/foo_tranche00_00001.json",
        sha256="f" * 64,
        member_count=1,
        logical_bytes=1,
    )
    for bad in ((0,), (-1,)):
        with pytest.raises(ValueError, match="positive integers"):
            MsaArtifactSetManifest(
                artifact_set_id=msa_artifact_set_id((chunk,), 1, 1, member_lengths=bad),
                chunks=(chunk,),
                member_count=1,
                logical_bytes=1,
                member_lengths=bad,
            )


def test_materialize_fails_closed_on_missing_backend_image() -> None:
    operational = _operational()
    selection = FoldingBackendImageSelection(
        backend_images={
            "openfold-cli": OPENFOLD_CLI_IMAGE,
            "bioir": "registry/bioir:latest",
            "colabfold": "registry/colabfold:latest",
        }
    )
    operational = FoldingPhaseAttemptOperationalSelection(
        profile_name=operational.profile_name,
        owner=operational.owner,
        transport=operational.transport,
        ssh_target=operational.ssh_target,
        account=operational.account,
        project_root=operational.project_root,
        staging_root=operational.staging_root,
        orchestration_repo=operational.orchestration_repo,
        runtime_image=operational.runtime_image,
        backend_images=selection,
        release_preset=_default_preset(),
        resources=operational.resources,
        fold_resources=operational.fold_resources,
        extra_mounts=operational.extra_mounts,
        assets=_assets("openfold-trt"),
    )
    with pytest.raises(ValueError, match="no kernel image selected"):
        materialize_folding_attempt_runspec(
            phase_run_id=PHASE_RUN_ID,
            attempt_id=ATTEMPT_ID,
            phase_plan=_phase_plan("openfold-trt"),
            materialized_at="2026-01-01T00:00:00Z",
            operational=operational,
        )


def _render(backend: str = "openfold-cli"):
    return render_folding_submission_intent(
        phase_runspec=_materialized(backend),
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )


def _container_mounts(script_body: str) -> str:
    match = re.search(r"--container-mounts=([^ ]+)", script_body)
    assert match is not None
    return match.group(1)


def test_render_returns_contract_payload() -> None:
    intent = _render()
    assert isinstance(intent, PhaseSubmissionIntendedPayload)
    reloaded = phase_submission_intended_payload_from_mapping(intent.to_mapping())
    assert reloaded == intent


def test_render_uses_bspp_phase_correlation_tokens() -> None:
    intent = _render()
    for plan in intent.actions:
        assert plan.scheduler_correlation_token.startswith("bspp-phase-")
        assert plan.job_name == plan.scheduler_correlation_token
        assert plan.scheduler_correlation_token == phase_action_scheduler_correlation_token(
            intent.submission_id, plan.action_id
        )
        assert plan.script_body.count(f"#SBATCH --job-name={plan.job_name}") == 1
        assert plan.script_body.count(f"#SBATCH --comment={plan.scheduler_correlation_token}") == 1


def test_render_qualification_tuple_id_is_deterministic() -> None:
    intent = _render()
    assert re.fullmatch(r"[0-9a-f]{64}", intent.qualification_tuple_id) is not None
    re_rendered = _render()
    assert re_rendered.submission_id == intent.submission_id
    assert re_rendered.qualification_tuple_id == intent.qualification_tuple_id
    assert [plan.script_body for plan in re_rendered.actions] == [plan.script_body for plan in intent.actions]
    bioir = _render("bioir")
    assert bioir.qualification_tuple_id != intent.qualification_tuple_id


def test_render_selects_kernel_image_by_backend() -> None:
    intent = _render()
    by_id = {plan.action_id: plan for plan in intent.actions}
    assert f"--container-image={OPENFOLD_CLI_IMAGE}" in by_id["fold-000001"].script_body
    for action_id in ACTION_IDS:
        if action_id != "fold-000001":
            assert f"--container-image={RUNTIME_IMAGE}" in by_id[action_id].script_body


def test_render_uses_executor_argv_without_embedded_python_c() -> None:
    intent = _render()
    assert [plan.action_id for plan in intent.actions] == ACTION_IDS
    by_id = {plan.action_id: plan for plan in intent.actions}
    for plan in intent.actions:
        assert plan.script_body.count("python -m bspp.orchestration.runtime.folding.executor") == 1
        assert "python -c" not in plan.script_body
        assert f"--action-id {plan.action_id}" in plan.script_body
        assert f"--handoff {plan.handoff_path}" in plan.script_body
        assert plan.handoff_path.endswith(f"/actions/{plan.action_id}/handoff.json")
        assert plan.script_body.count("--container-image=") == 1
        assert plan.script_body.count("srun --container-image=") == 1
        assert "pyxis" not in plan.script_body.lower()
        assert "enroot" not in plan.script_body.lower()
        assert "sbatch" not in plan.script_body
        assert "#SBATCH" in plan.script_body
    assert f"--container-image={OPENFOLD_CLI_IMAGE}" in by_id["fold-000001"].script_body
    for action_id in ACTION_IDS:
        if action_id != "fold-000001":
            assert f"--container-image={RUNTIME_IMAGE}" in by_id[action_id].script_body


def test_rendered_paths_satisfy_executor_path_validation() -> None:
    """Regression (council e09s03 dissent): the rendered --action-evidence and
    --handoff paths must pass the job-local executor's own path validators —
    handoff at <attempt-root>/actions/<action-id>/handoff.json and evidence
    confined beneath that action root — for every rendered action."""
    from pathlib import Path

    from bspp.orchestration.runtime.folding.executor import (
        _validate_evidence_path_confinement,
        _validate_handoff_path_identity,
    )

    intent = _render()
    for plan in intent.actions:
        action_root, _actions_dir, _attempt_root = _validate_handoff_path_identity(
            Path(plan.handoff_path), plan.action_id
        )
        _validate_evidence_path_confinement(Path(plan.action_evidence_path), action_root)
        assert Path(plan.action_evidence_path).parent == action_root


def test_render_documents_job_local_executor_seam() -> None:
    import inspect

    import bspp.orchestration.control.folding_phase_adapter as adapter

    module_doc = adapter.__doc__ or ""
    executor_doc = inspect.getsource(adapter._executor_argv)
    assert "job-local" in module_doc
    assert "job-local" in executor_doc
    assert "deferred" not in module_doc
    assert "deferred" not in executor_doc


def test_render_mounts_include_attempt_root_and_input_files_once() -> None:
    """Attempt root mounts rw once; each authenticated input FILE mounts :ro at
    its exact path, and the input parent DIRECTORY is never exposed."""
    intent = _render()
    attempt_root = f"/lustre/bspp/bspp-phase-runs/{PHASE_RUN_ID}/attempt-0001"
    tar = "/bspp-fixture/msa-set/msa-set.tar"
    bundle = "/bspp-fixture/msa-set/msa-set.tar.lz4"
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        assert mounts.count(f"{attempt_root}:{attempt_root}") == 1
        assert mounts.count(f"{tar}:{tar}:ro") == 1
        assert mounts.count(f"{bundle}:{bundle}:ro") == 1
        parent_mount = "/bspp-fixture/msa-set:/bspp-fixture/msa-set"
        assert parent_mount not in mounts


def test_render_mounts_honor_read_only_extra_mount() -> None:
    runspec = _materialized_with(
        extra_mounts=(PhaseMountSnapshot(source="/assets/models", target="/assets/models", read_only=True),)
    )
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        assert "/assets/models:/assets/models:ro" in mounts


def test_render_mounts_dedupe_identical_targets() -> None:
    tar = "/bspp-fixture/msa-set/msa-set.tar"
    runspec = _materialized_with(extra_mounts=(PhaseMountSnapshot(source=tar, target=tar, read_only=True),))
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        assert mounts.count(f"{tar}:{tar}:ro") == 1


def test_render_mounts_reject_conflicting_targets() -> None:
    tar = "/bspp-fixture/msa-set/msa-set.tar"
    runspec = _materialized_with(extra_mounts=(PhaseMountSnapshot(source=tar, target=tar, read_only=False),))
    with pytest.raises(ValueError, match="conflicting container mount target"):
        render_folding_submission_intent(
            phase_runspec=runspec,
            phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
            phase_runspec_document_sha256="e" * 64,
        )


def test_render_mounts_reject_pyxis_delimiters() -> None:
    """Greptile P1: mount values carrying the Pyxis list delimiter (comma), the
    source:target separator (colon), whitespace, or control characters must
    fail closed; everything else stays valid (the mount argument is quoted)."""
    for bad in (
        "/assets/with,comma",
        "/assets/with:colon",
        "/assets/with space",
        "/assets/with\nnewline",
        "relative/path",
    ):
        runspec = _materialized_with(
            extra_mounts=(PhaseMountSnapshot(source=bad, target="/assets/ok", read_only=True),)
        )
        with pytest.raises(ValueError, match="must be an absolute path free of"):
            render_folding_submission_intent(
                phase_runspec=runspec,
                phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
                phase_runspec_document_sha256="e" * 64,
            )


def test_render_mounts_accept_ordinary_filesystem_characters() -> None:
    """Paths with `+`, `@`, parentheses, or Unicode remain valid mounts."""
    runspec = _materialized_with(
        extra_mounts=(
            PhaseMountSnapshot(
                source="/assets/model+v2@(final)/éèmple", target="/assets/model+v2@(final)/éèmple", read_only=True
            ),
        )
    )
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        assert "/assets/model+v2@(final)/éèmple:/assets/model+v2@(final)/éèmple:ro" in mounts


def test_render_srun_quotes_container_arguments() -> None:
    """The srun container image and mount list are shell-quoted end to end."""
    intent = _render()
    for plan in intent.actions:
        srun = next(line for line in plan.script_body.splitlines() if line.startswith("srun "))
        assert " --container-mounts=" in srun
        assert " --container-image=" in srun
        # Allowlisted path characters need no quoting, so the flags stay bare;
        # the renderer must still pass both through shlex.quote.
        import inspect

        import bspp.orchestration.control.folding_phase_adapter as adapter

        assert "shlex.quote" in inspect.getsource(adapter._render_action_script)


def test_render_mounts_remote_input_keeps_extra_mounts_without_local_parent() -> None:
    runspec = _materialized_with(
        remote=True,
        extra_mounts=(PhaseMountSnapshot(source="/assets/models", target="/assets/models", read_only=True),),
    )
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        assert "/bspp-fixture/msa-set" not in mounts
        assert "/assets/models:/assets/models:ro" in mounts


def test_validate_evidence_builds_canonical_pair_index() -> None:
    index = validate_folding_action_evidence(phase_runspec=_materialized(), evidence=_full_evidence())
    assert index is not None
    assert index.schema_version == 1
    assert index.run_id == PHASE_RUN_ID
    assert len(index.entries) == 1
    entry = index.entries[0]
    assert entry.target_id == "target-1"
    assert entry.sequence_sha256 == "f" * 64
    assert entry.model_entity_id == "AF-0000000000000001"
    assert entry.tool_used == "OpenFold / AlphaFold-Multimer"


def test_validate_evidence_writes_index_loaded_by_track_c(tmp_path: Path) -> None:
    from bspp.orchestration.runtime.folding.benchmark.index import load_canonical_pair_index as track_c_load

    index_path = tmp_path / "canonical-pair-index.json"
    result = validate_folding_action_evidence(
        phase_runspec=_materialized(),
        evidence=_full_evidence(),
        index_path=index_path,
    )
    assert result is None
    track_c = track_c_load(index_path)
    assert track_c.schema_version == 1
    assert track_c.run_id == PHASE_RUN_ID
    assert track_c.to_mapping()["entries"][0]["target_id"] == "target-1"


def test_validate_rejects_evidence_not_covering_actions() -> None:
    evidence = _full_evidence()
    del evidence["fold-000001"]
    with pytest.raises(ValueError, match="cover exactly the RunSpec actions"):
        validate_folding_action_evidence(phase_runspec=_materialized(), evidence=evidence)


def test_validate_rejects_missing_canonical_pair() -> None:
    evidence = _full_evidence()
    evidence["fold-000001"] = {"pairs": []}
    with pytest.raises(ValueError, match="fold evidence pairs"):
        validate_folding_action_evidence(phase_runspec=_materialized(), evidence=evidence)


def test_validate_rejects_invalid_prediction_pair() -> None:
    bad_pair = _prediction_pair_mapping()
    scores = dict(bad_pair["scores"])
    scores["pae"] = [[0.0]]
    bad_pair["scores"] = scores
    evidence = _full_evidence()
    evidence["canonical-pair-000001"] = {
        "entries": [{"target_id": "target-1", "sequence_sha256": "f" * 64, "pair": bad_pair}]
    }
    with pytest.raises(ValueError):
        validate_folding_action_evidence(phase_runspec=_materialized(), evidence=evidence)


def test_validate_rejects_malformed_sequence_sha256() -> None:
    evidence = _full_evidence()
    evidence["canonical-pair-000001"] = {
        "entries": [{"target_id": "target-1", "sequence_sha256": "not-hex", "pair": _prediction_pair_mapping()}]
    }
    with pytest.raises(ValueError, match="sequence_sha256"):
        validate_folding_action_evidence(phase_runspec=_materialized(), evidence=evidence)


def test_validate_rejects_duplicate_target_id() -> None:
    pair = _prediction_pair_mapping()
    evidence = _full_evidence()
    evidence["canonical-pair-000001"] = {
        "entries": [
            {"target_id": "target-1", "sequence_sha256": "f" * 64, "pair": pair},
            {"target_id": "target-1", "sequence_sha256": "e" * 64, "pair": pair},
        ]
    }
    with pytest.raises(ValueError, match="Duplicate target_id"):
        validate_folding_action_evidence(phase_runspec=_materialized(), evidence=evidence)


def test_validate_rejects_unknown_tool_used() -> None:
    bad_pair = _prediction_pair_mapping()
    bad_pair["tool_used"] = "Nope"
    evidence = _full_evidence()
    evidence["canonical-pair-000001"] = {
        "entries": [{"target_id": "target-1", "sequence_sha256": "f" * 64, "pair": bad_pair}]
    }
    with pytest.raises(ValueError, match="tool_used"):
        validate_folding_action_evidence(phase_runspec=_materialized(), evidence=evidence)


def test_validate_rejects_canonical_pair_not_matching_fold() -> None:
    evidence = _full_evidence()
    different_pair = _prediction_pair_mapping("AF-0000000000000002")
    evidence["canonical-pair-000001"] = {
        "entries": [{"target_id": "target-1", "sequence_sha256": "f" * 64, "pair": different_pair}]
    }
    with pytest.raises(ValueError, match="exactly the fold action prediction pairs"):
        validate_folding_action_evidence(phase_runspec=_materialized(), evidence=evidence)


def test_validate_rejects_canonical_pair_content_mismatch() -> None:
    evidence = _full_evidence()
    mismatched = _prediction_pair_mapping()
    scores = dict(mismatched["scores"])
    scores["plddt"] = [0.9, 0.8]
    mismatched["scores"] = scores
    evidence["canonical-pair-000001"] = {
        "entries": [{"target_id": "target-1", "sequence_sha256": "f" * 64, "pair": mismatched}]
    }
    with pytest.raises(ValueError, match="does not match the fold action pair"):
        validate_folding_action_evidence(phase_runspec=_materialized(), evidence=evidence)


def test_control_side_index_surface_matches_track_c_fields() -> None:
    from bspp.orchestration.control.folding_phase_types import CanonicalPairIndex as ControlIndex
    from bspp.orchestration.control.folding_phase_types import CanonicalPairIndexEntry as ControlEntry
    from bspp.orchestration.runtime.folding.benchmark.index import CanonicalPairIndex as TrackCIndex
    from bspp.orchestration.runtime.folding.benchmark.index import CanonicalPairIndexEntry as TrackCEntry

    kwargs = {
        "target_id": "target-1",
        "sequence_sha256": "f" * 64,
        "model_entity_id": "AF-0000000000000001",
        "tool_used": "OpenFold / AlphaFold-Multimer",
        "structure_path": "AF-0000000000000001-model_v1.pdb",
        "scores_path": "AF-0000000000000001-meta_v1.json",
    }
    control_entry = ControlEntry(**kwargs)
    track_c_entry = TrackCEntry(**kwargs)
    assert set(control_entry.to_mapping()) == set(track_c_entry.to_mapping())
    control_index = ControlIndex(schema_version=1, run_id=PHASE_RUN_ID, entries=(control_entry,))
    track_c_index = TrackCIndex(schema_version=1, run_id=PHASE_RUN_ID, entries=(track_c_entry,))
    assert set(control_index.to_mapping()) == set(track_c_index.to_mapping())


# --- mount_orchestration_source opt-in mount tests ---


def _operational_with_mount(
    backend: str = "openfold-cli",
    *,
    mount_orchestration_source: bool = True,
    extra_mounts: tuple[PhaseMountSnapshot, ...] = (),
) -> FoldingPhaseAttemptOperationalSelection:
    op = _operational(backend, extra_mounts=extra_mounts)
    return replace(op, mount_orchestration_source=mount_orchestration_source)


def _materialized_with_mount(
    backend: str = "openfold-cli",
    *,
    mount_orchestration_source: bool = True,
    extra_mounts: tuple[PhaseMountSnapshot, ...] = (),
):
    return materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=_phase_plan(backend),
        materialized_at="2026-01-01T00:00:00Z",
        operational=_operational_with_mount(
            backend,
            mount_orchestration_source=mount_orchestration_source,
            extra_mounts=extra_mounts,
        ),
    )


def test_render_mounts_no_orchestration_source_by_default() -> None:
    """Default (mount_orchestration_source=False) does not emit the orch mount."""
    intent = _render()
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        assert "/workspace/bspp-orchestration" not in mounts


def test_render_mounts_include_orchestration_source_when_enabled() -> None:
    """When mount_orchestration_source=True, the orch mount is emitted."""
    runspec = _materialized_with_mount(mount_orchestration_source=True)
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    orch_repo = "/lustre/bspp/orchestration"
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        assert f"{orch_repo}:/workspace/bspp-orchestration:ro" in mounts


def test_render_srun_writable_only_when_mount_orchestration_source() -> None:
    """Override mode editable-installs into the container rootfs, so the srun
    must request --container-writable and the dev-mount env; baked mode stays
    read-only with no dev-mount env."""
    intent = _render()
    for plan in intent.actions:
        srun = next(line for line in plan.script_body.splitlines() if line.startswith("srun "))
        assert "--container-writable" not in srun
        assert "BSPP_ORCHESTRATION_DEV_MOUNT" not in srun

    runspec = _materialized_with_mount(mount_orchestration_source=True)
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    for plan in intent.actions:
        srun = next(line for line in plan.script_body.splitlines() if line.startswith("srun "))
        assert "--container-writable" in srun
        assert "BSPP_ORCHESTRATION_DEV_MOUNT=1" in srun


def test_render_mounts_orchestration_source_first() -> None:
    """The orchestration source mount appears before the RunSpec mount."""
    runspec = _materialized_with_mount(mount_orchestration_source=True)
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        orch_idx = mounts.find("/workspace/bspp-orchestration")
        runspec_idx = mounts.find("attempts/attempt-0001/phase-runspec.json")
        if runspec_idx == -1:
            # The RunSpec path may be different; find it by the mount pattern
            runspec_idx = mounts.find("phase-runspec.json")
        assert orch_idx != -1, "orchestration source mount not found"
        assert runspec_idx != -1, "runspec mount not found"
        assert orch_idx < runspec_idx, "orchestration source mount must come before RunSpec mount"


def test_render_mounts_orchestration_source_conflict_with_extra_mount() -> None:
    """An extra_mount targeting /workspace/bspp-orchestration with a different source is rejected."""
    runspec = _materialized_with_mount(
        mount_orchestration_source=True,
        extra_mounts=(
            PhaseMountSnapshot(source="/other/repo", target="/workspace/bspp-orchestration", read_only=True),
        ),
    )
    with pytest.raises(ValueError, match="conflicting container mount target"):
        render_folding_submission_intent(
            phase_runspec=runspec,
            phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
            phase_runspec_document_sha256="e" * 64,
        )


def test_render_mounts_orchestration_source_conflict_read_only_mismatch() -> None:
    """An extra_mount targeting /workspace/bspp-orchestration with read_only=False is rejected."""
    orch_repo = "/lustre/bspp/orchestration"
    runspec = _materialized_with_mount(
        mount_orchestration_source=True,
        extra_mounts=(PhaseMountSnapshot(source=orch_repo, target="/workspace/bspp-orchestration", read_only=False),),
    )
    with pytest.raises(ValueError, match="conflicting container mount target"):
        render_folding_submission_intent(
            phase_runspec=runspec,
            phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
            phase_runspec_document_sha256="e" * 64,
        )


def test_render_mounts_orchestration_source_dedup_identical() -> None:
    """An extra_mount with the same source and read_only=True at /workspace/bspp-orchestration deduplicates."""
    orch_repo = "/lustre/bspp/orchestration"
    runspec = _materialized_with_mount(
        mount_orchestration_source=True,
        extra_mounts=(PhaseMountSnapshot(source=orch_repo, target="/workspace/bspp-orchestration", read_only=True),),
    )
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    for plan in intent.actions:
        mounts = _container_mounts(plan.script_body)
        assert mounts.count(f"{orch_repo}:/workspace/bspp-orchestration:ro") == 1


def test_qualification_tuple_id_changes_when_mount_orchestration_source_flips() -> None:
    """Flipping mount_orchestration_source changes the cluster snapshot digest and qualification identity."""
    from bspp.orchestration.control.folding_phase_types import folding_qualification_tuple_id

    runspec_off = _materialized_with_mount(mount_orchestration_source=False)
    runspec_on = _materialized_with_mount(mount_orchestration_source=True)
    digest_off = canonical_mapping_digest(runspec_off.cluster.to_mapping())
    digest_on = canonical_mapping_digest(runspec_on.cluster.to_mapping())
    assert digest_off != digest_on
    tuple_off = folding_qualification_tuple_id(
        backend="openfold-cli",
        kernel_image=OPENFOLD_CLI_IMAGE,
        cluster_snapshot_digest=digest_off,
    )
    tuple_on = folding_qualification_tuple_id(
        backend="openfold-cli",
        kernel_image=OPENFOLD_CLI_IMAGE,
        cluster_snapshot_digest=digest_on,
    )
    assert tuple_off != tuple_on


# --- typed packed-topology resource tests (e13s02) ---


def test_phase_slurm_resources_typed_fields_round_trip() -> None:
    packed = PhaseSlurmResources(
        partition="gpu",
        cpus_per_task=30,
        memory="128G",
        time="04:00:00",
        nodes=2,
        tasks_per_node=8,
        gpus_per_task=1,
        max_parallel=2,
    )
    assert packed.workers == 16
    mapping = packed.to_mapping()
    assert mapping["nodes"] == 2
    assert mapping["tasks_per_node"] == 8
    assert mapping["gpus_per_task"] == 1
    assert mapping["max_parallel"] == 2

    scalar = PhaseSlurmResources(partition="gpu", cpus_per_task=30, memory="128G", time="04:00:00")
    assert scalar.workers == 1
    assert "tasks_per_node" not in scalar.to_mapping()
    assert "nodes" not in scalar.to_mapping()


def test_slurm_resources_typed_fields_round_trip() -> None:
    scalar = SlurmResources(partition="gpu", cpus_per_task=30, memory="128G", time="04:00:00")
    assert scalar.tasks_per_node == 1
    assert "tasks_per_node" not in scalar.model_dump()
    assert "nodes" not in scalar.model_dump()

    packed = SlurmResources(
        partition="gpu",
        cpus_per_task=30,
        memory="128G",
        time="04:00:00",
        nodes=2,
        tasks_per_node=8,
        gpus_per_task=1,
        max_parallel=2,
    )
    assert packed.nodes == 2
    assert packed.tasks_per_node == 8
    assert packed.gpus_per_task == 1
    assert packed.max_parallel == 2
    dumped = packed.model_dump()
    assert dumped["nodes"] == 2
    assert dumped["tasks_per_node"] == 8
    assert dumped["gpus_per_task"] == 1
    assert dumped["max_parallel"] == 2


def test_scalar_fold_action_resources_remain_byte_stable() -> None:
    runspec = _materialized()
    fold_action = runspec.payload.actions[3]
    mapping = fold_action.resources.to_mapping()
    assert "nodes" not in mapping
    assert "tasks_per_node" not in mapping
    assert "gpus_per_task" not in mapping
    assert "max_parallel" not in mapping
    assert mapping["gres"] == "gpu:1"
    assert mapping["array"] is None


def test_packed_fold_action_derives_array_and_workers() -> None:
    fold_resources = PhaseSlurmResources(
        partition="gpu",
        cpus_per_task=8,
        memory="64G",
        time="04:00:00",
        nodes=2,
        tasks_per_node=8,
        gpus_per_task=1,
        max_parallel=2,
        array="0-1%2",
    )
    operational = replace(_operational(), fold_resources=fold_resources)
    runspec = materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=_phase_plan(),
        materialized_at="2026-01-01T00:00:00Z",
        operational=operational,
    )
    fold_action = runspec.payload.actions[3]
    assert fold_action.resources.workers == 16
    assert fold_action.resources.array == "0-1%2"
    mapping = fold_action.resources.to_mapping()
    assert mapping["nodes"] == 2
    assert mapping["tasks_per_node"] == 8
    assert mapping["gpus_per_task"] == 1
    assert mapping["max_parallel"] == 2


# --- canonical fold shard projection tests (e13s05) ---


def test_scalar_materialization_produces_single_rank_projection_byte_stable() -> None:
    """A scalar fold topology (one worker) binds a single-rank projection that
    contains every target and whose document bytes are stable across repeated
    derivations."""
    runspec = _materialized()
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    assert binding.worker_count == 1
    assert binding.lpt_version == FOLD_SHARD_LPT_VERSION
    assert binding.location == f"attempts/{ATTEMPT_ID}/fold-shard-projection.json"

    fold_action = runspec.payload.actions[3]
    projection, expected_binding = derive_fold_shard_projection(_phase_plan(), fold_action.resources, ATTEMPT_ID)
    assert expected_binding == binding
    assert len(projection.ranks) == 1
    assert [target.target_id for target in projection.ranks[0].targets] == ["AF-0000000000000001"]

    document = fold_shard_projection_document_bytes(projection)
    assert hashlib.sha256(document).hexdigest() == binding.sha256
    assert len(document) == binding.size_bytes

    again_projection, again_binding = derive_fold_shard_projection(_phase_plan(), fold_action.resources, ATTEMPT_ID)
    assert again_binding == binding
    assert fold_shard_projection_document_bytes(again_projection) == document


def test_binding_validation_rejects_worker_count_drift() -> None:
    runspec = _materialized()
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    drifted = replace(binding, worker_count=binding.worker_count + 1)
    tampered = replace(runspec, payload=replace(runspec.payload, fold_shard_projection=drifted))
    with pytest.raises(ValueError, match="fold shard projection binding is not the canonical Plan-derived projection"):
        validate_folding_plan_runspec_binding(_phase_plan(), tampered)


def test_binding_validation_rejects_lpt_version_drift() -> None:
    runspec = _materialized()
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    drifted = replace(binding, lpt_version=binding.lpt_version + 1)
    tampered = replace(runspec, payload=replace(runspec.payload, fold_shard_projection=drifted))
    with pytest.raises(ValueError, match="fold shard projection binding is not the canonical Plan-derived projection"):
        validate_folding_plan_runspec_binding(_phase_plan(), tampered)


def test_materialize_phase_writes_fold_shard_projection_into_authority(tmp_path: Path) -> None:
    from tests.test_phase_folding_lifecycle import _materialize_folding

    authority_root, phase_run_id = _materialize_folding(tmp_path)
    run_root = authority_root / phase_run_id
    projection_path = run_root / "attempts" / "attempt-0001" / "fold-shard-projection.json"
    assert projection_path.is_file()
    assert not projection_path.is_symlink()

    validation = PhaseAuthorityStore(authority_root).validate(phase_run_id)
    assert isinstance(validation.phase_runspec, FoldingPhaseRunSpec)
    binding = validation.phase_runspec.payload.fold_shard_projection
    assert binding is not None
    assert binding.location == "attempts/attempt-0001/fold-shard-projection.json"

    document = projection_path.read_bytes()
    assert hashlib.sha256(document).hexdigest() == binding.sha256
    assert len(document) == binding.size_bytes

    expected_projection, expected_binding = fold_shard_projection_from_runspec(
        validation.phase_plan, validation.phase_runspec
    )
    assert expected_binding == binding
    assert document == fold_shard_projection_document_bytes(expected_projection)


# --- unified folding renderer tests (e13s06) ---


def _packed_fold_resources(*, gpus_per_task: int | None = 1, gres: str | None = "gpu:8") -> PhaseSlurmResources:
    return PhaseSlurmResources(
        partition="gpu",
        cpus_per_task=8,
        memory="64G",
        time="04:00:00",
        gres=gres,
        nodes=2,
        tasks_per_node=8,
        gpus_per_task=gpus_per_task,
        max_parallel=2,
    )


def _materialized_packed_fold(*, gpus_per_task: int | None = 1, gres: str | None = "gpu:8"):
    operational = replace(_operational(), fold_resources=_packed_fold_resources(gpus_per_task=gpus_per_task, gres=gres))
    return materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=_phase_plan(),
        materialized_at="2026-01-01T00:00:00Z",
        operational=operational,
    )


def test_render_packed_fold_emits_array_and_rank_formula() -> None:
    runspec = _materialized_packed_fold()
    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    by_id = {plan.action_id: plan for plan in intent.actions}
    fold_body = by_id["fold-000001"].script_body
    assert "#SBATCH --array=0-1%2" in fold_body
    assert "#SBATCH --nodes=1" in fold_body
    assert "#SBATCH --ntasks-per-node=8" in fold_body
    assert "#SBATCH --gpus-per-task=1" in fold_body
    assert "slurm-%A_%a.out" in fold_body
    assert "slurm-%A_%a.err" in fold_body
    assert "OPENFOLDCTL_GLOBAL_RANK=$((SLURM_ARRAY_TASK_ID * 8 + SLURM_PROCID))" in fold_body
    assert '--rank "${OPENFOLDCTL_GLOBAL_RANK}"' in fold_body
    assert fold_body.count("python -m bspp.orchestration.runtime.folding.executor") == 1
    assert "--ntasks=1" not in fold_body
    assert "%j" not in fold_body
    assert "--gres" not in fold_body

    # Non-fold actions stay scalar (single task, %j logs, no array).
    for action_id in ACTION_IDS:
        if action_id == "fold-000001":
            continue
        body = by_id[action_id].script_body
        assert "#SBATCH --ntasks=1" in body
        assert "slurm-%j.out" in body
        assert "--array" not in body
        assert "bash -c" not in body


def test_render_scalar_fold_has_no_array_and_single_srun() -> None:
    intent = _render()
    by_id = {plan.action_id: plan for plan in intent.actions}
    fold_body = by_id["fold-000001"].script_body
    assert "#SBATCH --nodes=1" in fold_body
    assert "#SBATCH --ntasks=1" in fold_body
    assert "slurm-%j.out" in fold_body
    assert "--array" not in fold_body
    assert "bash -c" not in fold_body
    assert fold_body.count("python -m bspp.orchestration.runtime.folding.executor") == 1
    assert fold_body.count("srun --container-image=") == 1


def test_render_packed_fold_requires_gpus_per_task_one() -> None:
    runspec = _materialized_packed_fold(gpus_per_task=2)
    with pytest.raises(ValueError, match="gpus-per-task=1"):
        render_folding_submission_intent(
            phase_runspec=runspec,
            phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
            phase_runspec_document_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tasks_per_node", 4),
        ("gpus_per_task", 1),
        ("max_parallel", 2),
    ],
)
def test_phase_slurm_resources_rejects_typed_fields_without_nodes(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="requires typed topology"):
        PhaseSlurmResources(
            partition="gpu",
            cpus_per_task=30,
            memory="128G",
            time="04:00:00",
            **{field: value},
        )


def test_phase_slurm_resources_rejects_typed_topology_without_gpus_per_task() -> None:
    with pytest.raises(ValueError, match="requires gpus_per_task"):
        PhaseSlurmResources(
            partition="gpu",
            cpus_per_task=30,
            memory="128G",
            time="04:00:00",
            nodes=2,
            tasks_per_node=8,
        )


def test_phase_slurm_resources_rejects_degenerate_typed_one_worker_topology() -> None:
    with pytest.raises(ValueError, match="degenerate one-worker shape"):
        PhaseSlurmResources(
            partition="gpu",
            cpus_per_task=30,
            memory="128G",
            time="04:00:00",
            nodes=1,
            tasks_per_node=1,
            gpus_per_task=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tasks_per_node", 4),
        ("gpus_per_task", 1),
        ("max_parallel", 2),
    ],
)
def test_slurm_resources_rejects_typed_fields_without_nodes(field: str, value: int) -> None:
    with pytest.raises(Exception, match="requires typed topology"):
        SlurmResources(
            partition="gpu",
            cpus_per_task=30,
            memory="128G",
            time="04:00:00",
            **{field: value},
        )


def test_slurm_resources_rejects_typed_topology_without_gpus_per_task() -> None:
    with pytest.raises(Exception, match="requires gpus_per_task"):
        SlurmResources(
            partition="gpu",
            cpus_per_task=30,
            memory="128G",
            time="04:00:00",
            nodes=2,
            tasks_per_node=8,
        )


def test_slurm_resources_rejects_degenerate_typed_one_worker_topology() -> None:
    with pytest.raises(Exception, match="degenerate one-worker shape"):
        SlurmResources(
            partition="gpu",
            cpus_per_task=30,
            memory="128G",
            time="04:00:00",
            nodes=1,
            tasks_per_node=1,
            gpus_per_task=1,
        )


@pytest.mark.parametrize("nodes,tasks_per_node", [(1, 8), (2, 1)])
def test_packed_worker_shapes_are_coherent_across_authorities(nodes: int, tasks_per_node: int) -> None:
    fold_resources = PhaseSlurmResources(
        partition="gpu",
        cpus_per_task=8,
        memory="64G",
        time="04:00:00",
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        gpus_per_task=1,
    )
    workers = nodes * tasks_per_node
    assert fold_resources.is_packed
    assert fold_resources.workers == workers

    operational = replace(_operational(), fold_resources=fold_resources)
    runspec = materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=_phase_plan(),
        materialized_at="2026-01-01T00:00:00Z",
        operational=operational,
    )
    fold_action = runspec.payload.actions[3]
    assert fold_action.resources.is_packed
    assert fold_action.resources.workers == workers
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    assert binding.worker_count == workers

    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    fold_plan = next(plan for plan in intent.actions if plan.action_id == "fold-000001")
    assert fold_plan.expected_task_indexes == tuple(range(nodes))
    assert f"#SBATCH --array=0-{nodes - 1}" in fold_plan.script_body
    assert f"#SBATCH --ntasks-per-node={tasks_per_node}" in fold_plan.script_body
    rank_formula = f"OPENFOLDCTL_GLOBAL_RANK=$((SLURM_ARRAY_TASK_ID * {tasks_per_node} + SLURM_PROCID))"
    assert rank_formula in fold_plan.script_body


def test_packed_materialization_omits_max_parallel_for_uncapped_array() -> None:
    fold_resources = PhaseSlurmResources(
        partition="gpu",
        cpus_per_task=8,
        memory="64G",
        time="04:00:00",
        nodes=3,
        tasks_per_node=2,
        gpus_per_task=1,
        max_parallel=None,
    )
    operational = replace(_operational(), fold_resources=fold_resources)
    runspec = materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=_phase_plan(),
        materialized_at="2026-01-01T00:00:00Z",
        operational=operational,
    )
    fold_action = runspec.payload.actions[3]
    assert fold_action.resources.max_parallel is None

    intent = render_folding_submission_intent(
        phase_runspec=runspec,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_document_sha256="e" * 64,
    )
    fold_plan = next(plan for plan in intent.actions if plan.action_id == "fold-000001")
    assert "#SBATCH --array=0-2" in fold_plan.script_body
    assert "%None" not in fold_plan.script_body
    assert "%" not in fold_plan.script_body.split("--array=0-2")[1].split("\n")[0]


def test_explicit_bioir_policy_binds_science_actions_and_read_only_assets() -> None:
    from bspp.orchestration.contract.folding_bioir import BioIRModelPolicy
    from bspp.orchestration.control.profiles import (
        FoldingBackendAssetsProfile,
        ProfileMount,
        validate_folding_backend_asset_mount_coverage,
    )

    policy = BioIRModelPolicy(
        monomer_checkpoint_sha256="a" * 64,
        monomer_checkpoint_size_bytes=100,
        multimer_checkpoint_sha256="b" * 64,
        multimer_checkpoint_size_bytes=200,
    )
    original = _phase_plan("bioir")
    plan = replace(original, payload=replace(original.payload, bioir_model_policy=policy))
    assets = FoldingBackendAssetsProfile(
        backend="bioir", bioir_checkpoint="/assets/multi.pt", bioir_monomer_checkpoint="/assets/mono.pt"
    ).to_snapshot()
    multi_mount = ProfileMount(source="/host/multi.pt", target="/assets/multi.pt", read_only=True)
    mono_mount = ProfileMount(source="/host/mono.pt", target="/assets/mono.pt", read_only=True)
    with pytest.raises(ValueError, match="bioir_monomer_checkpoint"):
        validate_folding_backend_asset_mount_coverage(assets, (multi_mount,))
    with pytest.raises(ValueError, match="bioir_monomer_checkpoint"):
        validate_folding_backend_asset_mount_coverage(
            assets, (multi_mount, mono_mount.model_copy(update={"read_only": False}))
        )
    validate_folding_backend_asset_mount_coverage(assets, (multi_mount, mono_mount))
    operational = replace(
        _operational("bioir"),
        assets=assets,
        extra_mounts=tuple(
            PhaseMountSnapshot(source=mount.source, target=mount.target, read_only=mount.read_only)
            for mount in (multi_mount, mono_mount)
        ),
    )
    runspec = materialize_folding_attempt_runspec(
        phase_run_id=PHASE_RUN_ID,
        attempt_id=ATTEMPT_ID,
        phase_plan=plan,
        materialized_at="2026-01-01T00:00:00Z",
        operational=operational,
    )
    assert runspec.payload.bioir_model_policy == policy
    validate_folding_plan_runspec_binding(plan, runspec)
    for action in runspec.payload.actions:
        params = dict(action.payload.params)
        if action.action_kind in {"fold", "canonical-pair"}:
            assert params["bioir_model_policy_digest"] == policy.digest
        else:
            assert "bioir_model_policy_digest" not in params
    with pytest.raises(ValueError, match="model policy"):
        validate_folding_plan_runspec_binding(original, runspec)
    bad_actions = tuple(
        replace(
            action,
            payload=replace(
                action.payload,
                params=tuple(
                    (key, "f" * 64 if key == "bioir_model_policy_digest" else value)
                    for key, value in action.payload.params
                ),
            ),
        )
        if action.action_kind == "fold"
        else action
        for action in runspec.payload.actions
    )
    with pytest.raises(ValueError, match="action graph"):
        validate_folding_plan_runspec_binding(
            plan, replace(runspec, payload=replace(runspec.payload, actions=bad_actions))
        )
