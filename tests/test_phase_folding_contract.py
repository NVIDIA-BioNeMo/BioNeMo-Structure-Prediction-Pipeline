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

"""Contract tests for the folding Phase envelope and family dispatch."""

from __future__ import annotations

import re
import typing
from dataclasses import replace

import pytest

from bspp.orchestration.contract.folding_evidence import (
    CanonicalPairActionEvidence,
    CanonicalPairEvidenceEntry,
    FoldActionEvidence,
    MsaFlattenActionEvidence,
    PreprocessActionEvidence,
    SplitActionEvidence,
)
from bspp.orchestration.contract.folding_execution import (
    FoldingBackendAssetsSnapshot,
    FoldingMsaFlattenHandoff,
    FoldingPreprocessTarget,
    FoldingTargetIdentity,
    folding_msa_flatten_handoff_from_mapping,
    folding_target_sequence_sha256,
)
from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.folding_release import FoldingReleasePreset
from bspp.orchestration.contract.phase import (
    FoldingActionPayload,
    FoldingPhasePlan,
    FoldingPhasePlanPayload,
    FoldingPhaseRunSpec,
    FoldingPhaseRunSpecPayload,
    FoldingResolvedClusterSnapshot,
    FoldingRuntimeAction,
    FoldingRuntimeActionKind,
    PhaseKind,
    PhaseMountSnapshot,
    PhaseSlurmResources,
    folding_phase_plan_payload_from_mapping,
    folding_phase_runspec_payload_from_mapping,
    folding_resolved_cluster_snapshot_from_mapping,
    phase_mount_snapshot_from_mapping,
    phase_plan_family_from_mapping,
    phase_runspec_family_from_mapping,
)
from bspp.orchestration.contract.phase_retry import (
    compare_retry_invariants,
    phase_input_set_identity_digest,
    phase_scientific_identity_digest,
)
from bspp.orchestration.contract.phase_state import (
    PhaseAttempt,
    PhaseMaterializedPayload,
    PhaseRun,
    phase_materialized_payload_from_mapping,
    phase_run_from_mapping,
)
from bspp.orchestration.contract.phase_submission import (
    _ACTION_ID,
    phase_action_scheduler_correlation_token,
)
from bspp.orchestration.contract.prediction_pair import prediction_pair_from_mapping
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    VerifiedLocalBundledArtifactLocation,
    VerifiedRemoteBundledArtifactLocation,
    msa_artifact_set_id,
    verified_local_bundled_artifact_location_id,
    verified_remote_bundled_artifact_location_id,
)
from bspp.orchestration.control.folding_phase_adapter import (
    materialize_folding_attempt_runspec,
    validate_folding_plan_runspec_binding,
)
from bspp.orchestration.control.folding_phase_types import (
    FoldingBackendImageSelection,
    FoldingPhaseAttemptOperationalSelection,
)

_MEMBER_NAME = "AFDB_AF-0000000000000000.a3m"


def _default_preset() -> FoldingReleasePreset:
    """The tree's current default folding release preset (INTERNAL on main, PUBLIC after sanitization)."""
    return next(iter(FoldingReleasePreset))


def make_msa_set(artifact_set_id: str = "sha256:" + "a" * 64) -> MsaSetConsumption:
    return MsaSetConsumption(
        artifact_set_id=artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(f"a3ms/{_MEMBER_NAME}",),
        requires_paired_query_header=True,
    )


def make_remote_location(artifact_set_id: str) -> VerifiedRemoteBundledArtifactLocation:
    member = BundledMemberVerification(
        logical_path=f"a3ms/{_MEMBER_NAME}",
        member_name=_MEMBER_NAME,
        raw_member_name=_MEMBER_NAME,
        size_bytes=123,
        sha256="f" * 64,
    )
    raw_tar_members = (_MEMBER_NAME,)
    members = (member,)
    bundle_uri = "s3://example-bucket-bucket/msa-sets/set.tar.lz4"
    location_id = verified_remote_bundled_artifact_location_id(
        artifact_set_id=artifact_set_id,
        bundle_uri=bundle_uri,
        tar_size_bytes=456,
        tar_sha256="e" * 64,
        lz4_size_bytes=789,
        lz4_sha256="d" * 64,
        raw_tar_members=raw_tar_members,
        members=members,
    )
    return VerifiedRemoteBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=artifact_set_id,
        bundle_uri=bundle_uri,
        tar_size_bytes=456,
        tar_sha256="e" * 64,
        lz4_size_bytes=789,
        lz4_sha256="d" * 64,
        raw_tar_members=raw_tar_members,
        members=members,
        verified_at="2026-09-11T00:00:00Z",
    )


def _make_local_location_mapping() -> dict[str, object]:
    """Create a valid VerifiedLocalBundledArtifactLocation mapping for handoff tests."""
    tar_path = "/bspp-fixture/msa-set/msa-set.tar"
    bundle_path = "/bspp-fixture/msa-set/msa-set.tar.lz4"
    bundle_uri = "file:///bspp-fixture/msa-set/msa-set.tar.lz4"
    members = (
        BundledMemberVerification(
            logical_path="a3ms/x.a3m",
            member_name="x.a3m",
            raw_member_name="x.a3m",
            size_bytes=1,
            sha256="b" * 64,
        ),
    )
    raw_tar_members = ("x.a3m",)
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id="sha256:" + "a" * 64,
        tar_path=tar_path,
        bundle_path=bundle_path,
        bundle_uri=bundle_uri,
        tar_size_bytes=1,
        tar_sha256="c" * 64,
        lz4_size_bytes=1,
        lz4_sha256="d" * 64,
        raw_tar_members=raw_tar_members,
        members=members,
    )
    location = VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id="sha256:" + "a" * 64,
        tar_path=tar_path,
        bundle_path=bundle_path,
        bundle_uri=bundle_uri,
        tar_size_bytes=1,
        tar_sha256="c" * 64,
        lz4_size_bytes=1,
        lz4_sha256="d" * 64,
        raw_tar_members=raw_tar_members,
        members=members,
        verified_at="2026-01-01T00:00:00Z",
    )
    return location.to_mapping()


def make_cluster() -> FoldingResolvedClusterSnapshot:
    return FoldingResolvedClusterSnapshot(
        profile_name="example-cluster-folding",
        owner="example-user",
        transport="ssh",
        ssh_target="example-cluster-oci-dc-02.example-cluster-oci-iad.nvidia.com",
        account="example-account",
        project_root="/srv/example/portfolios/example-account/projects/example-account/users/example-user/bspp",
        staging_root="/srv/example/portfolios/example-account/projects/example-account/users/example-user/bspp/staging",
        orchestration_repo=(
            "/srv/example/portfolios/example-account/projects/example-account/users/example-user/bspp/bspp-orchestration"
        ),
        runtime_image="registry.example.com/bspp:folding",
        extra_mounts=(),
    )


def make_resources() -> PhaseSlurmResources:
    return PhaseSlurmResources(partition="batch", cpus_per_task=8, memory="64G", time="04:00:00")


def make_action(
    action_id: str,
    action_kind: FoldingRuntimeActionKind,
    dependencies: tuple[str, ...] = (),
) -> FoldingRuntimeAction:
    return FoldingRuntimeAction(
        action_id=action_id,
        dependencies=dependencies,
        resources=make_resources(),
        payload=FoldingActionPayload(action_kind=action_kind, params=()),
        action_kind=action_kind,
    )


def make_actions() -> tuple[FoldingRuntimeAction, ...]:
    return (
        make_action("msa-flatten-000001", "msa-flatten"),
        make_action("split-000001", "split", ("msa-flatten-000001",)),
        make_action("preprocess-000001", "preprocess", ("split-000001",)),
        make_action("fold-000001", "fold", ("preprocess-000001",)),
        make_action("canonical-pair-000001", "canonical-pair", ("fold-000001",)),
    )


def make_plan() -> FoldingPhasePlan:
    msa_set = make_msa_set()
    return FoldingPhasePlan(
        target_cluster="example-cluster-folding",
        input_location=make_remote_location(msa_set.artifact_set_id),
        payload=FoldingPhasePlanPayload(msa_set=msa_set, backend="openfold-cli"),
    )


def make_runspec(plan: FoldingPhasePlan | None = None) -> FoldingPhaseRunSpec:
    if plan is None:
        plan = make_plan()
    return FoldingPhaseRunSpec(
        phase_run_id="phase-run-" + "b" * 32,
        attempt_id="attempt-0001",
        phase_plan_digest=plan.digest,
        materialized_at="2026-09-11T00:00:00Z",
        input_location=plan.input_location,
        cluster=make_cluster(),
        payload=FoldingPhaseRunSpecPayload(
            msa_set=plan.payload.msa_set,
            backend=plan.payload.backend,
            actions=make_actions(),
        ),
    )


def test_phase_kind_includes_folding() -> None:
    assert "folding" in typing.get_args(PhaseKind)
    assert "preprocessing" in typing.get_args(PhaseKind)


def test_folding_payloads_construct_round_trip_and_reject_unknown_backends() -> None:
    msa = make_msa_set()
    plan_payload = FoldingPhasePlanPayload(msa_set=msa, backend="openfold-cli")
    assert plan_payload.phase_kind == "folding"
    assert plan_payload.to_mapping()["backend"] == "openfold-cli"
    assert folding_phase_plan_payload_from_mapping(plan_payload.to_mapping()) == plan_payload

    runspec_payload = FoldingPhaseRunSpecPayload(msa_set=msa, backend="openfold-cli", actions=make_actions())
    assert folding_phase_runspec_payload_from_mapping(runspec_payload.to_mapping()) == runspec_payload

    with pytest.raises(ValueError, match="unsupported folding backend"):
        FoldingPhasePlanPayload(msa_set=msa, backend="bogus")
    with pytest.raises(ValueError, match="unsupported folding backend"):
        FoldingPhaseRunSpecPayload(msa_set=msa, backend="bogus", actions=make_actions())


def test_folding_action_graph_validation_matrix() -> None:
    FoldingPhaseRunSpecPayload(msa_set=make_msa_set(), backend="openfold-cli", actions=make_actions())

    cycle = (
        make_action("msa-flatten-000001", "msa-flatten", ("canonical-pair-000001",)),
        make_action("canonical-pair-000001", "canonical-pair", ("msa-flatten-000001",)),
    )
    with pytest.raises(ValueError, match="cycle"):
        FoldingPhaseRunSpecPayload(msa_set=make_msa_set(), backend="openfold-cli", actions=cycle)

    dangling = (make_action("fold-000001", "fold", ("missing-000001",)),)
    with pytest.raises(ValueError, match="dangling"):
        FoldingPhaseRunSpecPayload(msa_set=make_msa_set(), backend="openfold-cli", actions=dangling)

    with pytest.raises(ValueError, match="must match its"):
        make_action("fold-000001", "split")

    duplicate = (
        make_action("fold-000001", "fold"),
        make_action("fold-000001", "fold"),
    )
    with pytest.raises(ValueError, match="unique"):
        FoldingPhaseRunSpecPayload(msa_set=make_msa_set(), backend="openfold-cli", actions=duplicate)

    with pytest.raises(ValueError, match="payload kind must match"):
        FoldingRuntimeAction(
            action_id="fold-000001",
            dependencies=(),
            resources=make_resources(),
            payload=FoldingActionPayload(action_kind="split", params=()),
            action_kind="fold",
        )


def test_folding_plan_and_runspec_family_round_trip() -> None:
    plan = make_plan()
    loaded_plan = phase_plan_family_from_mapping(plan.to_mapping())
    assert loaded_plan == plan
    assert loaded_plan.digest == plan.digest
    assert re.fullmatch(r"[0-9a-f]{64}", plan.digest)

    runspec = make_runspec(plan)
    loaded_runspec = phase_runspec_family_from_mapping(runspec.to_mapping())
    assert loaded_runspec == runspec
    assert loaded_runspec.digest == runspec.digest


def test_folding_cluster_snapshot_round_trip() -> None:
    cluster = make_cluster()
    assert folding_resolved_cluster_snapshot_from_mapping(cluster.to_mapping()) == cluster

    with pytest.raises(ValueError, match="ssh_target must match"):
        FoldingResolvedClusterSnapshot(
            profile_name="p",
            owner="o",
            transport="local-slurm",
            ssh_target="example-cluster",
            account="a",
            project_root="/p",
            staging_root="/s",
            orchestration_repo="/o",
            runtime_image="img",
            extra_mounts=(),
        )


def test_submission_action_id_acceptance_matrix() -> None:
    submission_id = "phase-submission-" + "a" * 64
    for action_id in (
        "msa-flatten-000001",
        "split-000001",
        "preprocess-000001",
        "fold-000001",
        "canonical-pair-000001",
        "preprocessing-chunk-000001",
    ):
        assert phase_action_scheduler_correlation_token(submission_id, action_id).startswith("bspp-phase-")

    for bad in ("bogus-000001", "fold-00001", "fold-0000001"):
        assert _ACTION_ID.fullmatch(bad) is None
        with pytest.raises(ValueError, match="invalid Runtime Action identity"):
            phase_action_scheduler_correlation_token(submission_id, bad)


def test_retry_identity_digest_family_dispatch() -> None:
    plan = make_plan()
    input_digest = phase_input_set_identity_digest(plan)
    scientific_digest = phase_scientific_identity_digest(plan)
    assert re.fullmatch(r"[0-9a-f]{64}", input_digest)
    assert re.fullmatch(r"[0-9a-f]{64}", scientific_digest)
    assert input_digest != scientific_digest

    other_backend = FoldingPhasePlan(
        target_cluster=plan.target_cluster,
        input_location=plan.input_location,
        payload=FoldingPhasePlanPayload(msa_set=plan.payload.msa_set, backend="bioir"),
    )
    assert phase_input_set_identity_digest(other_backend) != input_digest

    other_msa = make_msa_set("sha256:" + "b" * 64)
    other_plan = FoldingPhasePlan(
        target_cluster=plan.target_cluster,
        input_location=make_remote_location(other_msa.artifact_set_id),
        payload=FoldingPhasePlanPayload(msa_set=other_msa, backend="openfold-cli"),
    )
    assert phase_input_set_identity_digest(other_plan) != input_digest


def test_compare_retry_invariants_folding_dispatch() -> None:
    plan = make_plan()
    predecessor = make_runspec(plan)
    successor = replace(predecessor, attempt_id="attempt-0002", materialized_at="2026-09-11T01:00:00Z")
    compare_retry_invariants(plan, predecessor, successor)

    changed = replace(successor, payload=replace(successor.payload, backend="bioir"))
    with pytest.raises(ValueError, match="non-allowlisted"):
        compare_retry_invariants(plan, predecessor, changed)


def test_family_dispatchers_reject_unknown_phase_kind() -> None:
    with pytest.raises(ValueError, match="unsupported Phase Plan phase_kind"):
        phase_plan_family_from_mapping({"phase_kind": "bogus"})
    with pytest.raises(ValueError, match="unsupported Phase RunSpec phase_kind"):
        phase_runspec_family_from_mapping({"phase_kind": "bogus"})


def test_phase_run_reconstructs_with_folding_kind() -> None:
    attempt = PhaseAttempt(
        attempt_id="attempt-0001",
        ordinal=1,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_digest="a" * 64,
        created_at="2026-09-11T00:00:00Z",
    )
    run = PhaseRun(
        phase_run_id="phase-run-" + "b" * 32,
        phase_plan_location="phase-plan.json",
        phase_plan_digest="a" * 64,
        created_at="2026-09-11T00:00:00Z",
        current_attempt_id="attempt-0001",
        attempts=(attempt,),
        phase_kind="folding",
    )
    assert run.phase_kind == "folding"
    assert phase_run_from_mapping(run.to_mapping()) == run


def test_materialized_payload_accepts_folding_runspec() -> None:
    plan = make_plan()
    runspec = make_runspec(plan)
    attempt = PhaseAttempt(
        attempt_id="attempt-0001",
        ordinal=1,
        phase_runspec_location="attempts/attempt-0001/phase-runspec.json",
        phase_runspec_digest=runspec.digest,
        created_at=runspec.materialized_at,
    )
    run = PhaseRun(
        phase_run_id=runspec.phase_run_id,
        phase_plan_location="phase-plan.json",
        phase_plan_digest=plan.digest,
        created_at=runspec.materialized_at,
        current_attempt_id="attempt-0001",
        attempts=(attempt,),
        phase_kind="folding",
    )
    payload = PhaseMaterializedPayload(phase_run=run, phase_runspec=runspec)
    assert phase_materialized_payload_from_mapping(payload.to_mapping()) == payload


def _prediction_pair_mapping(model_entity_id: str = "AF-0000000000000001") -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_entity_id": model_entity_id,
        "tool_used": "OpenFold / AlphaFold-Multimer",
        "structure_path": f"{model_entity_id}-model_v1.pdb",
        "scores_path": f"{model_entity_id}-meta_v1.json",
        "scores": {
            "schema_version": 1,
            "plddt": [0.5, 0.6],
            "pae": [[0.0, 0.1], [0.1, 0.0]],
            "max_pae": 0.1,
            "ptm": 0.9,
            "iptm": 0.8,
        },
    }


def _folding_target_identity() -> FoldingTargetIdentity:
    return FoldingTargetIdentity(
        target_id="target-1",
        description="description",
        chains=("ACD",),
        sequence_sha256=folding_target_sequence_sha256(("ACD",)),
    )


def test_folding_evidence_strict_round_trip() -> None:
    pair_mapping = _prediction_pair_mapping()
    pair = prediction_pair_from_mapping(pair_mapping)
    entry_mapping = {"target_id": "t", "sequence_sha256": "f" * 64, "pair": pair_mapping}

    assert MsaFlattenActionEvidence.from_mapping({"a3m_paths": ["a3ms/x.a3m"]}) == MsaFlattenActionEvidence(
        a3m_paths=("a3ms/x.a3m",)
    )
    assert SplitActionEvidence.from_mapping({"chain_files": ["chain_1.a3m"]}) == SplitActionEvidence(
        chain_files=("chain_1.a3m",)
    )
    assert PreprocessActionEvidence.from_mapping(
        {"fasta_dir": "fasta", "alignment_dir": "alignments", "layout": "openfold"}
    ) == PreprocessActionEvidence(fasta_dir="fasta", alignment_dir="alignments", layout="openfold")
    assert FoldActionEvidence.from_mapping({"pairs": [pair_mapping]}) == FoldActionEvidence(pairs=(pair,))
    assert CanonicalPairEvidenceEntry.from_mapping(entry_mapping) == CanonicalPairEvidenceEntry(
        target_id="t", sequence_sha256="f" * 64, pair=pair
    )
    assert CanonicalPairActionEvidence.from_mapping({"entries": [entry_mapping]}) == CanonicalPairActionEvidence(
        entries=(CanonicalPairEvidenceEntry(target_id="t", sequence_sha256="f" * 64, pair=pair),)
    )


def test_folding_evidence_rejects_unknown_fields() -> None:
    loaders = (
        MsaFlattenActionEvidence.from_mapping,
        SplitActionEvidence.from_mapping,
        PreprocessActionEvidence.from_mapping,
        FoldActionEvidence.from_mapping,
        CanonicalPairEvidenceEntry.from_mapping,
        CanonicalPairActionEvidence.from_mapping,
    )
    for loader in loaders:
        with pytest.raises(ValueError, match="Unknown"):
            loader({"extra": "nope"})


def test_folding_evidence_control_reexport_identity() -> None:
    from bspp.orchestration.control.folding_phase_types import (
        CanonicalPairActionEvidence as ControlCanonicalPairActionEvidence,
    )
    from bspp.orchestration.control.folding_phase_types import (
        CanonicalPairEvidenceEntry as ControlCanonicalPairEvidenceEntry,
    )
    from bspp.orchestration.control.folding_phase_types import (
        FoldActionEvidence as ControlFoldActionEvidence,
    )
    from bspp.orchestration.control.folding_phase_types import (
        MsaFlattenActionEvidence as ControlMsaFlattenActionEvidence,
    )
    from bspp.orchestration.control.folding_phase_types import (
        PreprocessActionEvidence as ControlPreprocessActionEvidence,
    )
    from bspp.orchestration.control.folding_phase_types import (
        SplitActionEvidence as ControlSplitActionEvidence,
    )

    assert ControlMsaFlattenActionEvidence is MsaFlattenActionEvidence
    assert ControlSplitActionEvidence is SplitActionEvidence
    assert ControlPreprocessActionEvidence is PreprocessActionEvidence
    assert ControlFoldActionEvidence is FoldActionEvidence
    assert ControlCanonicalPairEvidenceEntry is CanonicalPairEvidenceEntry
    assert ControlCanonicalPairActionEvidence is CanonicalPairActionEvidence


def test_phase_mount_snapshot_read_only_default_and_round_trip() -> None:
    mount = PhaseMountSnapshot(source="/s", target="/t")
    assert mount.read_only is False
    assert "read_only" not in mount.to_mapping()

    loaded = phase_mount_snapshot_from_mapping({"source": "/s", "target": "/t"})
    assert loaded.read_only is False
    assert loaded == mount

    read_only = phase_mount_snapshot_from_mapping({"source": "/s", "target": "/t", "read_only": True})
    assert read_only.read_only is True
    assert read_only.to_mapping()["read_only"] is True
    assert PhaseMountSnapshot(source="/s", target="/t", read_only=True) == read_only

    with pytest.raises(ValueError, match="read_only must be a boolean"):
        phase_mount_snapshot_from_mapping({"source": "/s", "target": "/t", "read_only": "yes"})


def test_colabfold_layout_acceptance() -> None:
    identity = _folding_target_identity()
    for layout in ("openfold", "bioir", "colabfold"):
        target = FoldingPreprocessTarget(
            target=identity,
            layout=layout,
            fasta_dir="/w/fasta",
            alignment_dir="/w/alignments",
            template_dir="/w/templates",
        )
        assert target.layout == layout

    with pytest.raises(ValueError, match="unsupported preprocess target layout"):
        FoldingPreprocessTarget(
            target=identity,
            layout="bogus",
            fasta_dir="/w/fasta",
            alignment_dir="/w/alignments",
            template_dir="/w/templates",
        )

    for layout in ("openfold", "bioir", "colabfold"):
        evidence = PreprocessActionEvidence.from_mapping(
            {"fasta_dir": "/w/fasta", "alignment_dir": "/w/alignments", "layout": layout}
        )
        assert evidence.layout == layout

    with pytest.raises(ValueError, match="layout"):
        PreprocessActionEvidence.from_mapping(
            {"fasta_dir": "/w/fasta", "alignment_dir": "/w/alignments", "layout": "bogus"}
        )


# ---------------------------------------------------------------------------
# Backend assets on FoldingPhaseAttemptOperationalSelection + binding
# ---------------------------------------------------------------------------


def make_manifest() -> MsaArtifactSetManifest:
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


def make_plan_with_manifest() -> FoldingPhasePlan:
    manifest = make_manifest()
    msa_set = MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(f"a3ms/{_MEMBER_NAME}",),
        requires_paired_query_header=True,
    )
    return FoldingPhasePlan(
        target_cluster="example-cluster-folding",
        input_location=make_remote_location(manifest.artifact_set_id),
        payload=FoldingPhasePlanPayload(msa_set=msa_set, backend="openfold-cli", msa_set_manifest=manifest),
    )


def _operational_selection(assets: FoldingBackendAssetsSnapshot | None) -> FoldingPhaseAttemptOperationalSelection:
    return FoldingPhaseAttemptOperationalSelection(
        profile_name="example-cluster-folding",
        owner="example-user",
        transport="ssh",
        ssh_target="example-cluster-oci-dc-02.example-cluster-oci-iad.nvidia.com",
        account="example-account",
        project_root="/p",
        staging_root="/s",
        orchestration_repo="/o",
        runtime_image="registry.example.com/bspp:folding",
        backend_images=FoldingBackendImageSelection(
            backend_images={"openfold-cli": "registry.example.com/openfold:latest"}
        ),
        release_preset=_default_preset(),
        resources=make_resources(),
        assets=assets,
    )


def _openfold_cli_assets() -> FoldingBackendAssetsSnapshot:
    return FoldingBackendAssetsSnapshot(
        backend="openfold-cli",
        chain_manifest_csv="/assets/chains.csv",
        openfold_model_dir="/assets/models",
    )


def test_operational_selection_assets_validation() -> None:
    assets = _openfold_cli_assets()
    assert _operational_selection(assets).assets == assets
    assert _operational_selection(None).assets is None
    with pytest.raises(ValueError, match="assets must be a FoldingBackendAssetsSnapshot"):
        FoldingPhaseAttemptOperationalSelection(
            profile_name="p",
            owner="o",
            transport="ssh",
            ssh_target="example-cluster",
            account="a",
            project_root="/p",
            staging_root="/s",
            orchestration_repo="/o",
            runtime_image="img",
            backend_images=FoldingBackendImageSelection(backend_images={"openfold-cli": "img"}),
            release_preset=_default_preset(),
            resources=make_resources(),
            assets="not-a-snapshot",
        )


def test_binding_reconstructs_assets_from_cluster_snapshot() -> None:
    assets = _openfold_cli_assets()
    plan = make_plan_with_manifest()
    runspec = materialize_folding_attempt_runspec(
        phase_run_id="phase-run-" + "b" * 32,
        attempt_id="attempt-0001",
        phase_plan=plan,
        materialized_at="2026-09-11T00:00:00Z",
        operational=_operational_selection(assets),
    )
    assert runspec.cluster.backend_assets == assets
    validate_folding_plan_runspec_binding(plan, runspec)


def test_binding_tolerates_none_backend_assets() -> None:
    plan = make_plan_with_manifest()
    runspec = materialize_folding_attempt_runspec(
        phase_run_id="phase-run-" + "b" * 32,
        attempt_id="attempt-0001",
        phase_plan=plan,
        materialized_at="2026-09-11T00:00:00Z",
        operational=_operational_selection(_openfold_cli_assets()),
    )
    pre_field = replace(runspec, cluster=replace(runspec.cluster, backend_assets=None))
    validate_folding_plan_runspec_binding(plan, pre_field)


def test_binding_rejects_assets_tagged_for_another_backend() -> None:
    """Regression (council e09s02 dissent): a validly-shaped assets snapshot
    tagged for a different backend must not bind the selected backend's RunSpec."""
    plan = make_plan_with_manifest()
    runspec = materialize_folding_attempt_runspec(
        phase_run_id="phase-run-" + "b" * 32,
        attempt_id="attempt-0001",
        phase_plan=plan,
        materialized_at="2026-09-11T00:00:00Z",
        operational=_operational_selection(_openfold_cli_assets()),
    )
    wrong_backend = FoldingBackendAssetsSnapshot(backend="openfold-trt", chain_manifest_csv="/assets/chains.csv")
    drifted = replace(runspec, cluster=replace(runspec.cluster, backend_assets=wrong_backend))
    with pytest.raises(ValueError, match="backend assets do not bind the selected backend"):
        validate_folding_plan_runspec_binding(plan, drifted)


def test_binding_rejects_msa_set_manifest_drift() -> None:
    plan = make_plan_with_manifest()
    runspec = materialize_folding_attempt_runspec(
        phase_run_id="phase-run-" + "b" * 32,
        attempt_id="attempt-0001",
        phase_plan=plan,
        materialized_at="2026-09-11T00:00:00Z",
        operational=_operational_selection(_openfold_cli_assets()),
    )
    assert runspec.payload.msa_set_manifest == plan.payload.msa_set_manifest
    drifted = replace(runspec, payload=replace(runspec.payload, msa_set_manifest=None))
    with pytest.raises(ValueError, match="msa_set_manifest does not bind"):
        validate_folding_plan_runspec_binding(plan, drifted)


# --- mount_orchestration_source snapshot tests ---


def test_folding_resolved_cluster_snapshot_mount_orchestration_source_default() -> None:
    """Loading a snapshot without mount_orchestration_source defaults to False."""
    cluster = make_cluster()
    mapping = cluster.to_mapping()
    assert "mount_orchestration_source" not in mapping
    loaded = folding_resolved_cluster_snapshot_from_mapping(mapping)
    assert loaded.mount_orchestration_source is False


def test_folding_resolved_cluster_snapshot_mount_orchestration_source_true() -> None:
    """Loading a snapshot with mount_orchestration_source: true sets the field to True."""
    cluster = replace(make_cluster(), mount_orchestration_source=True)
    mapping = cluster.to_mapping()
    assert mapping["mount_orchestration_source"] is True
    loaded = folding_resolved_cluster_snapshot_from_mapping(mapping)
    assert loaded.mount_orchestration_source is True


def test_folding_resolved_cluster_snapshot_mount_orchestration_source_str_coercion_rejected() -> None:
    """Loading a snapshot with a string 'true' raises ValueError (strict isinstance)."""
    cluster = replace(make_cluster(), mount_orchestration_source=True)
    mapping = cluster.to_mapping()
    mapping["mount_orchestration_source"] = "true"
    with pytest.raises(ValueError, match="mount_orchestration_source must be a boolean"):
        folding_resolved_cluster_snapshot_from_mapping(mapping)


def test_folding_resolved_cluster_snapshot_mount_orchestration_source_serialized() -> None:
    """to_mapping includes mount_orchestration_source only when True."""
    cluster_off = make_cluster()
    assert "mount_orchestration_source" not in cluster_off.to_mapping()
    cluster_on = replace(make_cluster(), mount_orchestration_source=True)
    assert cluster_on.to_mapping()["mount_orchestration_source"] is True


def test_folding_resolved_cluster_snapshot_post_init_rejects_non_bool() -> None:
    """Direct construction with a non-bool raises TypeError (defense-in-depth)."""
    with pytest.raises(TypeError, match="mount_orchestration_source must be a bool"):
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
            mount_orchestration_source="true",  # type: ignore[arg-type]
        )


# --- Handoff provenance field tests ---


def test_handoff_install_mode_field_optional() -> None:
    """Loading a handoff without install_mode succeeds."""
    local_location = _make_local_location_mapping()
    handoff = FoldingMsaFlattenHandoff(
        phase_run_id="phase-run-" + "a" * 32,
        attempt_id="attempt-0001",
        action_id="msa-flatten-000001",
        predecessor_digest=None,
        projected_members=(("a3ms/x.a3m", "/tmp/x.a3m"),),
        local_location=local_location,
    )
    mapping = handoff.to_mapping()
    assert "install_mode" not in mapping
    assert "orchestration_source_commit" not in mapping
    loaded = folding_msa_flatten_handoff_from_mapping(mapping)
    assert loaded.install_mode is None
    assert loaded.orchestration_source_commit is None


def test_handoff_install_mode_field_validated() -> None:
    """Loading a handoff with install_mode='invalid' raises ValueError."""
    local_location = _make_local_location_mapping()
    handoff_mapping = {
        "schema_version": 1,
        "phase_run_id": "phase-run-" + "a" * 32,
        "attempt_id": "attempt-0001",
        "action_id": "msa-flatten-000001",
        "predecessor_digest": None,
        "projected_members": [["a3ms/x.a3m", "/tmp/x.a3m"]],
        "local_location": local_location,
        "install_mode": "invalid",
    }
    with pytest.raises(ValueError, match="install_mode must be"):
        folding_msa_flatten_handoff_from_mapping(handoff_mapping)


def test_handoff_install_mode_field_in_to_mapping() -> None:
    """to_mapping includes install_mode when set, omits when None."""
    local_location = _make_local_location_mapping()
    handoff = FoldingMsaFlattenHandoff(
        phase_run_id="phase-run-" + "a" * 32,
        attempt_id="attempt-0001",
        action_id="msa-flatten-000001",
        predecessor_digest=None,
        projected_members=(("a3ms/x.a3m", "/tmp/x.a3m"),),
        local_location=local_location,
        install_mode="override",
        orchestration_source_commit="abc123",
    )
    mapping = handoff.to_mapping()
    assert mapping["install_mode"] == "override"
    assert mapping["orchestration_source_commit"] == "abc123"


def test_handoff_empty_source_commit_rejected() -> None:
    """Loading a handoff with orchestration_source_commit='' raises ValueError."""
    local_location = _make_local_location_mapping()
    handoff_mapping = {
        "schema_version": 1,
        "phase_run_id": "phase-run-" + "a" * 32,
        "attempt_id": "attempt-0001",
        "action_id": "msa-flatten-000001",
        "predecessor_digest": None,
        "projected_members": [["a3ms/x.a3m", "/tmp/x.a3m"]],
        "local_location": local_location,
        "orchestration_source_commit": "",
    }
    with pytest.raises(ValueError, match="must be null or a non-empty string"):
        folding_msa_flatten_handoff_from_mapping(handoff_mapping)
