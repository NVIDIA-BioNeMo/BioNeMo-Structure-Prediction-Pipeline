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

"""Executor tests: run the five folding actions over real filesystem fixtures.

The MSA projection/transport seam uses the real local ``prepare_folding_msa_input``
(no lz4 subprocess), the split/preprocess seams use the real file-staging helpers,
and only the three scientific fold backends are mocked at the boundary. The five
dispatchers run over a real tar + strict chain-manifest CSV, and the terminal
canonical-pair action is accepted by the Control-side
``validate_folding_action_evidence`` validator.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_execution import (
    OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR,
    FoldingBackendAssetsSnapshot,
    FoldingMsaFlattenHandoff,
    folding_canonical_pair_handoff_from_mapping,
    folding_fold_handoff_from_mapping,
    folding_msa_flatten_handoff_from_mapping,
    folding_preprocess_handoff_from_mapping,
    folding_split_handoff_from_mapping,
)
from bspp.orchestration.contract.folding_input import MsaSetConsumption
from bspp.orchestration.contract.folding_shard import (
    FoldShardProjection,
    FoldShardProjectionBinding,
    FoldShardRank,
    FoldShardTarget,
    fold_shard_projection_document_bytes,
)
from bspp.orchestration.contract.model_identity import normalize_model_entity_id
from bspp.orchestration.contract.phase import (
    FoldingActionPayload,
    FoldingPhaseRunSpec,
    FoldingPhaseRunSpecPayload,
    FoldingResolvedClusterSnapshot,
    FoldingRuntimeAction,
    PhaseSlurmResources,
    canonical_mapping_digest,
)
from bspp.orchestration.contract.prediction_pair import (
    PredictionPair,
    prediction_scores_payload_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    BundledMemberVerification,
    MsaArtifactSetManifest,
    MsaChunkManifestReference,
    VerifiedLocalBundledArtifactLocation,
    msa_artifact_set_id,
    verified_local_bundled_artifact_location_id,
)
from bspp.orchestration.control.folding_phase_adapter import validate_folding_action_evidence
from bspp.orchestration.runtime.folding.benchmark.index import build_canonical_pair_index, load_canonical_pair_index
from bspp.orchestration.runtime.folding.execution.a3m_split import split_merged_a3m
from bspp.orchestration.runtime.folding.execution.bioir_inputs import BioIRInputPreprocessor
from bspp.orchestration.runtime.folding.execution.chain_manifest import parse_chain_manifest
from bspp.orchestration.runtime.folding.execution.colabfold_inputs import prepare as colabfold_prepare
from bspp.orchestration.runtime.folding.execution.emitter_support import canonical_pair_names, serialize_scores_json
from bspp.orchestration.runtime.folding.execution.models import (
    FoldingResult,
    PreparedInput,
    ProteinTarget,
    StructurePrediction,
)
from bspp.orchestration.runtime.folding.execution.msa_intake import prepare_folding_msa_input
from bspp.orchestration.runtime.folding.execution.openfold_inputs import OpenFoldInputPreprocessor
from bspp.orchestration.runtime.folding.executor import (
    PRODUCTION_EXECUTOR_DEPS,
    FoldingExecutorDependencies,
    FoldingExecutorError,
    _with_install_provenance,
    run_execute_action,
)
from bspp.orchestration.runtime.folding.rank_journal import folding_qualification_tuple_id, read_rank_journal


@pytest.fixture(autouse=True)
def _clear_install_provenance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure provenance env vars are unset for hermetic executor tests."""
    monkeypatch.delenv("BSPP_ORCHESTRATION_SOURCE", raising=False)
    monkeypatch.delenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", raising=False)


_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "folding" / "executable"
_PHASE_RUN_ID = "phase-run-" + "a" * 32
_ATTEMPT_ID = "attempt-0001"
_VERIFIED_AT = "2026-09-01T00:00:00.000000Z"
_MEMBER_NAME = "AFDB_AF-0000000000000001_AF-0000000000000002.a3m"
_TARGET_ID = "AF-0000000000000001_AF-0000000000000002"
_LOGICAL_PATH = f"a3ms/{_MEMBER_NAME}"
_SECOND_TARGET_ID = "AF-0000000000000003_AF-0000000000000004"
_SECOND_LOGICAL_PATH = "a3ms/AFDB_AF-0000000000000003_AF-0000000000000004.a3m"
_ACTION_IDS = {
    "msa-flatten": "msa-flatten-000001",
    "split": "split-000002",
    "preprocess": "preprocess-000003",
    "fold": "fold-000004",
    "canonical-pair": "canonical-pair-000005",
}
_OPENFOLD_TOOL_USED = "OpenFold / AlphaFold-Multimer"
_COLABFOLD_TOOL_USED = "ColabFold v1.6.0 / AlphaFold-Multimer"
_BIOIR_TOOL_USED = "OpenFold2 (BioNeMo IR) / AlphaFold-Multimer"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_tar(entries: list[tuple[str, bytes]]) -> tuple[bytes, str, int, tuple[str, ...]]:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        root = tarfile.TarInfo(".")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        raw_names = tuple(header.name for header in archive.getmembers())
    return raw, _sha256(raw), len(raw), raw_names


def _make_manifest(member_count: int) -> MsaArtifactSetManifest:
    chunk_name = "sample_tranche00_00001.fa"
    reference = MsaChunkManifestReference(
        chunk_name=chunk_name,
        logical_path=f"chunks/{chunk_name.removesuffix('.fa')}.json",
        sha256="a" * 64,
        member_count=member_count,
        logical_bytes=member_count * 100,
    )
    return MsaArtifactSetManifest(
        artifact_set_id=msa_artifact_set_id((reference,), member_count, member_count * 100),
        chunks=(reference,),
        member_count=member_count,
        logical_bytes=member_count * 100,
    )


def _make_consumption(manifest: MsaArtifactSetManifest) -> MsaSetConsumption:
    return MsaSetConsumption(
        artifact_set_id=manifest.artifact_set_id,
        expected_chunk_count=1,
        member_a3m_paths=(_LOGICAL_PATH,),
        requires_paired_query_header=True,
    )


def _make_local_location(
    tmp_path: Path,
    manifest: MsaArtifactSetManifest,
    member_bytes: bytes,
) -> VerifiedLocalBundledArtifactLocation:
    tar_path = (tmp_path / "bundle.tar").absolute()
    tar_bytes, _, _, raw_names = _write_tar([(f"./{_MEMBER_NAME}", member_bytes)])
    tar_path.write_bytes(tar_bytes)
    bundle_path = (tmp_path / "bundle.tar.lz4").absolute()
    bundle_bytes = b"not-a-real-lz4-stream"
    bundle_path.write_bytes(bundle_bytes)
    member = BundledMemberVerification(
        logical_path=_LOGICAL_PATH,
        member_name=_MEMBER_NAME,
        raw_member_name=f"./{_MEMBER_NAME}",
        size_bytes=len(member_bytes),
        sha256=_sha256(member_bytes),
    )
    location_id = verified_local_bundled_artifact_location_id(
        artifact_set_id=manifest.artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_path.as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=_sha256(tar_bytes),
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=_sha256(bundle_bytes),
        raw_tar_members=raw_names,
        members=(member,),
    )
    return VerifiedLocalBundledArtifactLocation(
        artifact_location_id=location_id,
        artifact_set_id=manifest.artifact_set_id,
        tar_path=str(tar_path),
        bundle_path=str(bundle_path),
        bundle_uri=bundle_path.as_uri(),
        tar_size_bytes=len(tar_bytes),
        tar_sha256=_sha256(tar_bytes),
        lz4_size_bytes=len(bundle_bytes),
        lz4_sha256=_sha256(bundle_bytes),
        raw_tar_members=raw_names,
        members=(member,),
        verified_at=_VERIFIED_AT,
    )


def _make_backend_assets(backend: str, chain_manifest_csv: Path) -> FoldingBackendAssetsSnapshot:
    if backend == "openfold-cli":
        return FoldingBackendAssetsSnapshot(
            backend="openfold-cli",
            chain_manifest_csv=str(chain_manifest_csv),
            openfold_model_dir="/models/openfold",
        )
    if backend == "colabfold":
        return FoldingBackendAssetsSnapshot(
            backend="colabfold",
            chain_manifest_csv=str(chain_manifest_csv),
            colabfold_weights_dir="/models/colabfold",
        )
    if backend == "bioir":
        return FoldingBackendAssetsSnapshot(backend="bioir", bioir_checkpoint="/models/bioir/checkpoint.pt")
    if backend == "openfold-trt":
        return FoldingBackendAssetsSnapshot(backend="openfold-trt", chain_manifest_csv=str(chain_manifest_csv))
    raise AssertionError(f"unsupported test backend: {backend}")


def _make_runspec(
    tmp_path: Path,
    *,
    backend: str,
    chain_manifest_csv: Path,
) -> tuple[FoldingPhaseRunSpec, VerifiedLocalBundledArtifactLocation]:
    manifest = _make_manifest(1)
    consumption = _make_consumption(manifest)
    location = _make_local_location(tmp_path, manifest, (_FIXTURE_DIR / "merged_compound.a3m").read_bytes())
    resources = PhaseSlurmResources(partition="gpu", cpus_per_task=1, memory="1G", time="00:10:00")
    actions = tuple(
        FoldingRuntimeAction(
            action_id=_ACTION_IDS[kind],
            dependencies=() if kind == "msa-flatten" else (_ACTION_IDS[previous],),
            resources=resources,
            payload=FoldingActionPayload(action_kind=kind, params=()),
            action_kind=kind,
        )
        for kind, previous in (
            ("msa-flatten", None),
            ("split", "msa-flatten"),
            ("preprocess", "split"),
            ("fold", "preprocess"),
            ("canonical-pair", "fold"),
        )
    )
    cluster = FoldingResolvedClusterSnapshot(
        profile_name="acceptance",
        owner="tester",
        transport="local-slurm",
        ssh_target=None,
        account="account",
        project_root="/project",
        staging_root="/staging",
        orchestration_repo="/repo",
        runtime_image="/image.sqsh",
        extra_mounts=(),
        backend_assets=_make_backend_assets(backend, chain_manifest_csv),
    )
    payload = FoldingPhaseRunSpecPayload(
        msa_set=consumption,
        backend=backend,
        actions=actions,
        msa_set_manifest=manifest,
    )
    runspec = FoldingPhaseRunSpec(
        phase_run_id=_PHASE_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        phase_plan_digest="a" * 64,
        materialized_at=_VERIFIED_AT,
        input_location=location,
        cluster=cluster,
        payload=payload,
    )
    return runspec, location


def _staged_projection_path(tmp_path: Path) -> Path:
    """Return the canonical projection path beside the staged RunSpec fixture.

    The executor resolves ``binding.location``'s basename against the directory of
    the ``--phase-runspec`` path it is given; every executor test passes
    ``tmp_path / "runspec.json"``, so the projection must sit at
    ``tmp_path / "fold-shard-projection.json"``.
    """
    return tmp_path / "fold-shard-projection.json"


def _with_fold_shard_binding(runspec: FoldingPhaseRunSpec, binding: FoldShardProjectionBinding) -> FoldingPhaseRunSpec:
    """Return a copy of the RunSpec with a different fold shard projection binding."""
    return FoldingPhaseRunSpec(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_plan_digest=runspec.phase_plan_digest,
        materialized_at=runspec.materialized_at,
        input_location=runspec.input_location,
        cluster=runspec.cluster,
        payload=FoldingPhaseRunSpecPayload(
            msa_set=runspec.payload.msa_set,
            backend=runspec.payload.backend,
            actions=runspec.payload.actions,
            msa_set_manifest=runspec.payload.msa_set_manifest,
            fold_shard_projection=binding,
        ),
        carry_forward=runspec.carry_forward,
    )


def _make_packed_runspec(
    tmp_path: Path,
    *,
    backend: str,
    chain_manifest_csv: Path,
    worker_count: int,
    rank_target_ids: dict[int, tuple[str, ...]],
    lpt_version: int = 1,
    nodes: int | None = None,
    tasks_per_node: int | None = None,
) -> tuple[FoldingPhaseRunSpec, VerifiedLocalBundledArtifactLocation]:
    """Build a packed fold RunSpec whose fold action carries a kernel_image param
    and a digest-bound canonical shard projection written beside the staged RunSpec."""
    manifest = _make_manifest(1)
    consumption = _make_consumption(manifest)
    location = _make_local_location(tmp_path, manifest, (_FIXTURE_DIR / "merged_compound.a3m").read_bytes())
    resources = PhaseSlurmResources(partition="gpu", cpus_per_task=1, memory="1G", time="00:10:00")
    if worker_count > 1:
        fold_nodes = nodes if nodes is not None else worker_count
        fold_tpn = tasks_per_node if tasks_per_node is not None else 1
        fold_resources = PhaseSlurmResources(
            partition="gpu",
            cpus_per_task=1,
            memory="1G",
            time="00:10:00",
            nodes=fold_nodes,
            tasks_per_node=fold_tpn,
            gpus_per_task=1,
        )
    else:
        fold_resources = resources
    actions = tuple(
        FoldingRuntimeAction(
            action_id=_ACTION_IDS[kind],
            dependencies=() if kind == "msa-flatten" else (_ACTION_IDS[previous],),
            resources=fold_resources if kind == "fold" else resources,
            payload=FoldingActionPayload(
                action_kind=kind,
                params=(("backend", backend), ("kernel_image", "/images/kernel.sqsh")) if kind == "fold" else (),
            ),
            action_kind=kind,
        )
        for kind, previous in (
            ("msa-flatten", None),
            ("split", "msa-flatten"),
            ("preprocess", "split"),
            ("fold", "preprocess"),
            ("canonical-pair", "fold"),
        )
    )
    staging_root = tmp_path / "staging"
    cluster = FoldingResolvedClusterSnapshot(
        profile_name="acceptance",
        owner="tester",
        transport="local-slurm",
        ssh_target=None,
        account="account",
        project_root="/project",
        staging_root=str(staging_root),
        orchestration_repo="/repo",
        runtime_image="/image.sqsh",
        extra_mounts=(),
        backend_assets=_make_backend_assets(backend, chain_manifest_csv),
    )
    projection = FoldShardProjection(
        worker_count=worker_count,
        lpt_version=lpt_version,
        ranks=tuple(
            FoldShardRank(
                global_rank=rank,
                targets=tuple(
                    FoldShardTarget(target_id=target_id, member_length=100) for target_id in rank_target_ids[rank]
                ),
            )
            for rank in range(worker_count)
        ),
    )
    document = fold_shard_projection_document_bytes(projection)
    binding = FoldShardProjectionBinding(
        location=f"attempts/{_ATTEMPT_ID}/fold-shard-projection.json",
        sha256=_sha256(document),
        size_bytes=len(document),
        worker_count=worker_count,
        lpt_version=lpt_version,
    )
    _staged_projection_path(tmp_path).write_bytes(document)
    payload = FoldingPhaseRunSpecPayload(
        msa_set=consumption,
        backend=backend,
        actions=actions,
        msa_set_manifest=manifest,
        fold_shard_projection=binding,
    )
    runspec = FoldingPhaseRunSpec(
        phase_run_id=_PHASE_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        phase_plan_digest="a" * 64,
        materialized_at=_VERIFIED_AT,
        input_location=location,
        cluster=cluster,
        payload=payload,
    )
    return runspec, location


def _run_action(
    tmp_path: Path,
    runspec: FoldingPhaseRunSpec,
    deps: FoldingExecutorDependencies,
    action_id: str,
    *,
    rank: int = 0,
) -> tuple[Path, dict[str, object]]:
    action_root = tmp_path / "attempt" / "actions" / action_id
    action_root.mkdir(parents=True, exist_ok=True)
    handoff = run_execute_action(
        phase_runspec_path=tmp_path / "runspec.json",
        action_id=action_id,
        action_evidence_path=action_root / "action-evidence.json",
        handoff_path=action_root / "handoff.json",
        deps=deps,
        rank=rank,
    )
    return action_root, handoff


class RecordingOpenFoldBackend:
    name = "openfold-cli"

    def __init__(self, model_dir: Path, manifest: object) -> None:
        self.model_dir = model_dir
        self.manifest = manifest
        self.runs: list[tuple[ProteinTarget, PreparedInput, Path, bool]] = []

    def run(
        self,
        target: ProteinTarget,
        prepared: PreparedInput,
        output_dir: Path,
        *,
        leaked_homodimer: bool = False,
    ) -> FoldingResult:
        self.runs.append((target, prepared, output_dir, leaked_homodimer))
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_name, scores_name = canonical_pair_names(target.target_id)
        structure_path = output_dir / structure_name
        scores_path = output_dir / scores_name
        structure_path.write_text("FAKE PDB\n", encoding="utf-8")
        scores_path.write_text(
            serialize_scores_json(plddt=[1.0, 2.0, 3.0, 4.0], pae=[[0.1] * 4 for _ in range(4)], max_pae=0.1),
            encoding="utf-8",
        )
        return FoldingResult(
            backend="openfold-cli",
            predictions=(StructurePrediction(rank=1, structure_path=structure_path, scores_path=scores_path),),
            metadata={"tool_used": _OPENFOLD_TOOL_USED, "model_preset": "model_1_multimer_v3"},
        )


class RecordingColabFoldBackend:
    name = "colabfold"

    def __init__(self, config: object, manifest: object) -> None:
        self.config = config
        self.manifest = manifest
        self.runs: list[tuple[ProteinTarget, PreparedInput, Path]] = []

    def run(self, target: ProteinTarget, prepared: PreparedInput, output_dir: Path) -> FoldingResult:
        self.runs.append((target, prepared, output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_name, scores_name = canonical_pair_names(target.target_id)
        structure_path = output_dir / structure_name
        scores_path = output_dir / scores_name
        structure_path.write_text("FAKE PDB\n", encoding="utf-8")
        scores_path.write_text(
            serialize_scores_json(plddt=[1.0, 2.0, 3.0, 4.0], pae=[[0.1] * 4 for _ in range(4)], max_pae=0.1),
            encoding="utf-8",
        )
        return FoldingResult(
            backend="colabfold",
            predictions=(StructurePrediction(rank=1, structure_path=structure_path, scores_path=scores_path),),
            metadata={"tool_used": _COLABFOLD_TOOL_USED},
        )


class RecordingBioIRSession:
    name = "bioir"

    def __init__(self, checkpoint: Path, output_dir: Path) -> None:
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.run_calls: list[tuple[ProteinTarget, PreparedInput, Path]] = []
        self.closed = False

    def run(self, target: ProteinTarget, prepared: PreparedInput, output_dir: Path) -> FoldingResult:
        self.run_calls.append((target, prepared, output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        structure_name, scores_name = canonical_pair_names(target.target_id)
        structure_path = self.output_dir / structure_name
        scores_path = self.output_dir / scores_name
        structure_path.write_text("FAKE PDB\n", encoding="utf-8")
        scores_path.write_text(
            serialize_scores_json(plddt=[1.0, 2.0, 3.0, 4.0], pae=[[0.1] * 4 for _ in range(4)], max_pae=0.1),
            encoding="utf-8",
        )
        return FoldingResult(
            backend="bioir",
            predictions=(StructurePrediction(rank=1, structure_path=structure_path, scores_path=scores_path),),
            metadata={"tool_used": _BIOIR_TOOL_USED},
        )

    def close(self) -> None:
        self.closed = True


class _RecordingDeps:
    def __init__(self, runspec: FoldingPhaseRunSpec, backend: str) -> None:
        self.runspec = runspec
        self.backend = backend
        self.lz4_argv: tuple[str, ...] | None = None
        self.openfold_backends: list[RecordingOpenFoldBackend] = []
        self.colabfold_backends: list[RecordingColabFoldBackend] = []
        self.bioir_sessions: list[RecordingBioIRSession] = []

    def load_runspec(self, path: Path) -> FoldingPhaseRunSpec:
        return self.runspec

    def msa_intake(
        self,
        runspec: FoldingPhaseRunSpec,
        workspace: Path,
        *,
        artifact_set: object,
        lz4_argv: tuple[str, ...],
    ) -> tuple[object, dict[str, Path]]:
        self.lz4_argv = lz4_argv
        return prepare_folding_msa_input(runspec, workspace, artifact_set=artifact_set, lz4_argv=lz4_argv)

    def openfold_backend_factory(self, model_dir: Path, manifest: object) -> RecordingOpenFoldBackend:
        backend = RecordingOpenFoldBackend(model_dir, manifest)
        self.openfold_backends.append(backend)
        return backend

    def colabfold_backend_factory(self, config: object, manifest: object) -> RecordingColabFoldBackend:
        backend = RecordingColabFoldBackend(config, manifest)
        self.colabfold_backends.append(backend)
        return backend

    def bioir_session_factory(self, checkpoint: Path, output_dir: Path) -> RecordingBioIRSession:
        session = RecordingBioIRSession(checkpoint, output_dir)
        self.bioir_sessions.append(session)
        return session

    def deps(self) -> FoldingExecutorDependencies:
        return FoldingExecutorDependencies(
            load_runspec=self.load_runspec,
            msa_intake=self.msa_intake,
            split_merged=split_merged_a3m,
            openfold_preprocess=lambda: OpenFoldInputPreprocessor(),
            bioir_preprocess=lambda: BioIRInputPreprocessor(),
            colabfold_prepare=colabfold_prepare,
            openfold_backend_factory=self.openfold_backend_factory,
            colabfold_backend_factory=self.colabfold_backend_factory,
            bioir_session_factory=self.bioir_session_factory,
            parse_chain_manifest=parse_chain_manifest,
        )


def _run_graph(
    tmp_path: Path,
    runspec: FoldingPhaseRunSpec,
    deps: FoldingExecutorDependencies,
) -> tuple[dict[str, object], Path, FoldingExecutorError | None]:
    attempt_root = tmp_path / "attempt"
    outcomes: dict[str, object] = {}
    error: FoldingExecutorError | None = None
    for action in runspec.payload.actions:
        action_root = attempt_root / "actions" / action.action_id
        action_root.mkdir(parents=True, exist_ok=True)
        handoff_path = action_root / "handoff.json"
        evidence_path = action_root / "action-evidence.json"
        try:
            handoff = run_execute_action(
                phase_runspec_path=tmp_path / "runspec.json",
                action_id=action.action_id,
                action_evidence_path=evidence_path,
                handoff_path=handoff_path,
                deps=deps,
            )
        except FoldingExecutorError as exc:
            outcomes[action.action_id] = {
                "handoff_path": handoff_path,
                "evidence_path": evidence_path,
                "handoff": None,
                "error": exc,
            }
            error = exc
            break
        outcomes[action.action_id] = {
            "handoff_path": handoff_path,
            "evidence_path": evidence_path,
            "handoff": handoff,
            "error": None,
        }
    return outcomes, attempt_root, error


def _read_handoff(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_openfold_cli_graph_publishes_and_validates(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    outcomes, attempt_root, error = _run_graph(tmp_path, runspec, recording.deps())

    assert error is None
    assert recording.lz4_argv == ("/usr/bin/lz4", "-d", "-c")

    msa_flatten = folding_msa_flatten_handoff_from_mapping(
        _read_handoff(outcomes[_ACTION_IDS["msa-flatten"]]["handoff_path"])
    )
    split = folding_split_handoff_from_mapping(_read_handoff(outcomes[_ACTION_IDS["split"]]["handoff_path"]))
    preprocess = folding_preprocess_handoff_from_mapping(
        _read_handoff(outcomes[_ACTION_IDS["preprocess"]]["handoff_path"])
    )
    fold = folding_fold_handoff_from_mapping(_read_handoff(outcomes[_ACTION_IDS["fold"]]["handoff_path"]))
    canonical = folding_canonical_pair_handoff_from_mapping(
        _read_handoff(outcomes[_ACTION_IDS["canonical-pair"]]["handoff_path"])
    )

    # Exact backend config/path args.
    assert len(recording.openfold_backends) == 1
    backend = recording.openfold_backends[0]
    assert backend.model_dir == Path("/models/openfold")
    assert len(backend.manifest.rows) == 2
    # Not-leaked heterodimer classification from the strict CSV.
    assert len(backend.runs) == 1
    assert backend.runs[0][3] is False

    # Predecessor handoff digest progression.
    assert split.predecessor_digest == canonical_mapping_digest(msa_flatten.to_mapping())
    assert preprocess.predecessor_digest == canonical_mapping_digest(split.to_mapping())
    assert fold.predecessor_digest == canonical_mapping_digest(preprocess.to_mapping())
    assert canonical.predecessor_digest == canonical_mapping_digest(fold.to_mapping())

    # Canonical PredictionPair bytes.
    assert len(fold.targets) == 1
    pair = fold.targets[0].pair
    expected_pair = PredictionPair(
        model_entity_id=normalize_model_entity_id(_TARGET_ID),
        tool_used=_OPENFOLD_TOOL_USED,
        structure_path=pair.structure_path,
        scores_path=pair.scores_path,
        scores=prediction_scores_payload_from_mapping(json.loads(Path(pair.scores_path).read_text(encoding="utf-8"))),
    )
    assert pair == expected_pair

    # Final index bytes: round-trip and digest binds the written artifact.
    index_path = attempt_root / "actions" / _ACTION_IDS["canonical-pair"] / "canonical-pair-index.json"
    loaded_index = load_canonical_pair_index(index_path)
    assert loaded_index.run_id == runspec.phase_run_id
    assert canonical.index_digest == _sha256(index_path.read_bytes())
    assert canonical.index_digest == _sha256(loaded_index.to_json().encode())

    # Golden index fixture matches the written artifact after root normalization.
    golden = (_FIXTURE_DIR / "expected-canonical-pair-index.json").read_text(encoding="utf-8")
    assert golden.replace("/ATTEMPT_ROOT", str(attempt_root)) == index_path.read_text(encoding="utf-8")

    # Golden evidence fixtures match the emitted leaf records.
    assert json.loads(outcomes[_ACTION_IDS["msa-flatten"]]["evidence_path"].read_text()) == json.loads(
        (_FIXTURE_DIR / "expected-msa-flatten-evidence.json").read_text()
    )
    assert json.loads(outcomes[_ACTION_IDS["split"]]["evidence_path"].read_text()) == json.loads(
        (_FIXTURE_DIR / "expected-split-evidence.json").read_text()
    )
    assert json.loads(outcomes[_ACTION_IDS["preprocess"]]["evidence_path"].read_text()) == json.loads(
        (_FIXTURE_DIR / "expected-preprocess-evidence.json").read_text()
    )

    # The combined canonical-pair evidence is accepted by the Control validator.
    combined = json.loads(outcomes[_ACTION_IDS["canonical-pair"]]["evidence_path"].read_text())
    expected_index = build_canonical_pair_index(
        run_id=runspec.phase_run_id,
        entries=[
            (
                pair.model_entity_id,
                fold.targets[0].target.sequence_sha256,
                pair.model_entity_id,
                pair.tool_used,
                pair.structure_path,
                pair.scores_path,
            )
        ],
    )
    validated_index = validate_folding_action_evidence(phase_runspec=runspec, evidence=combined, index_path=None)
    assert validated_index.to_mapping() == expected_index.to_mapping()


def test_openfold_cli_leaked_homodimer_classification(tmp_path: Path) -> None:
    chain_manifest_csv = tmp_path / "leaked.csv"
    chain_manifest_csv.write_text(
        "model_entity_id,entity_id,chain_id,uniprot_ac\n"
        f"{_TARGET_ID},entity-A,chain-A,P00001\n"
        f"{_TARGET_ID},entity-B,chain-B,P00001\n",
        encoding="utf-8",
    )
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    _, _, error = _run_graph(tmp_path, runspec, recording.deps())

    assert error is None
    assert len(recording.openfold_backends) == 1
    assert recording.openfold_backends[0].runs[0][3] is True


def test_openfold_cli_ambiguous_target_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = tmp_path / "ambiguous.csv"
    chain_manifest_csv.write_text(
        "model_entity_id,entity_id,chain_id,uniprot_ac\n"
        "AF-0000000000000009_AF-0000000000000009,entity-A,chain-A,P00001\n",
        encoding="utf-8",
    )
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    outcomes, _, error = _run_graph(tmp_path, runspec, recording.deps())

    assert error is not None
    assert "ambiguous" in str(error)
    fold_outcome = outcomes[_ACTION_IDS["fold"]]
    assert not fold_outcome["handoff_path"].exists()
    assert not fold_outcome["evidence_path"].exists()


def test_colabfold_backend_config_and_empty_predictions_fail_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="colabfold", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="colabfold")
    outcomes, _, error = _run_graph(tmp_path, runspec, recording.deps())

    assert error is None
    assert len(recording.colabfold_backends) == 1
    backend = recording.colabfold_backends[0]
    assert backend.config.weights_dir == Path("/models/colabfold")
    assert backend.config.msa_cache_dir.name == "colabfold-cache"
    assert backend.config.structures_dir.name == "colabfold-structures"
    preprocess_evidence = json.loads(outcomes[_ACTION_IDS["preprocess"]]["evidence_path"].read_text())
    assert preprocess_evidence["layout"] == "colabfold"


def test_bioir_session_reuse_and_close(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="bioir", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="bioir")
    _, _, error = _run_graph(tmp_path, runspec, recording.deps())

    assert error is None
    assert len(recording.bioir_sessions) == 1
    session = recording.bioir_sessions[0]
    assert session.checkpoint == Path("/models/bioir/checkpoint.pt")
    assert len(session.run_calls) == 1
    assert session.closed is True


def test_openfold_trt_fails_closed_before_publication(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-trt", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-trt")
    outcomes, _, error = _run_graph(tmp_path, runspec, recording.deps())

    assert error is not None
    assert str(error) == OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR
    fold_outcome = outcomes[_ACTION_IDS["fold"]]
    assert not fold_outcome["handoff_path"].exists()
    assert not fold_outcome["evidence_path"].exists()


def test_backend_factory_failure_never_publishes_success(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")

    def failing_factory(model_dir: Path, manifest: object) -> RecordingOpenFoldBackend:
        raise FoldingExecutorError("scientific backend failed to initialize")

    deps = recording.deps()
    deps = FoldingExecutorDependencies(
        load_runspec=deps.load_runspec,
        msa_intake=deps.msa_intake,
        split_merged=deps.split_merged,
        openfold_preprocess=deps.openfold_preprocess,
        bioir_preprocess=deps.bioir_preprocess,
        colabfold_prepare=deps.colabfold_prepare,
        openfold_backend_factory=failing_factory,
        colabfold_backend_factory=deps.colabfold_backend_factory,
        bioir_session_factory=deps.bioir_session_factory,
        parse_chain_manifest=deps.parse_chain_manifest,
    )
    outcomes, _, error = _run_graph(tmp_path, runspec, deps)

    assert error is not None
    fold_outcome = outcomes[_ACTION_IDS["fold"]]
    assert not fold_outcome["handoff_path"].exists()
    assert not fold_outcome["evidence_path"].exists()


def test_rerun_guard_rejects_completed_action(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    deps = recording.deps()
    action_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["msa-flatten"]
    action_root.mkdir(parents=True, exist_ok=True)
    handoff_path = action_root / "handoff.json"
    evidence_path = action_root / "action-evidence.json"

    run_execute_action(
        phase_runspec_path=tmp_path / "runspec.json",
        action_id=_ACTION_IDS["msa-flatten"],
        action_evidence_path=evidence_path,
        handoff_path=handoff_path,
        deps=deps,
    )
    with pytest.raises(FoldingExecutorError, match="rerun"):
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["msa-flatten"],
            action_evidence_path=evidence_path,
            handoff_path=handoff_path,
            deps=deps,
        )


def test_rerun_guard_rejects_partial_content_but_allows_slurm_logs(tmp_path: Path) -> None:
    """Greptile P1: an interrupted action (outputs written, handoff/evidence never
    published) must fail closed on rerun — only scheduler logs may pre-exist."""
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    deps = recording.deps()
    action_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["msa-flatten"]
    stray = action_root / "projected" / "stale-member.a3m"
    stray.parent.mkdir(parents=True)
    stray.write_text(">stale\nAAA\n", encoding="utf-8")

    with pytest.raises(FoldingExecutorError, match="not fresh"):
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["msa-flatten"],
            action_evidence_path=action_root / "action-evidence.json",
            handoff_path=action_root / "handoff.json",
            deps=deps,
        )

    # Scheduler logs from the current job are the only allowed pre-existing content.
    import bspp.orchestration.runtime.folding.executor as executor

    stray.unlink()
    stray.parent.rmdir()
    (action_root / "slurm-123.out").write_text("log", encoding="utf-8")
    (action_root / "slurm-123.err").write_text("", encoding="utf-8")
    executor._reject_partial_rerun_content(action_root, rank=0, packed=False)  # must not raise
    (action_root / "slurm-123.out").unlink()
    (action_root / "slurm-123.err").unlink()


def test_split_evidence_uses_actual_per_target_filenames(tmp_path: Path) -> None:
    """Greptile P1: multi-target split evidence must name the files actually
    produced (each target's split restarts at chain_1.a3m), never a global
    enumeration of nonexistent files."""
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    deps = recording.deps()

    # Run the real flatten action, then widen its handoff to a second member
    # (same verified bytes, distinct logical stem -> second target).
    flatten_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["msa-flatten"]
    flatten_root.mkdir(parents=True)
    run_execute_action(
        phase_runspec_path=tmp_path / "runspec.json",
        action_id=_ACTION_IDS["msa-flatten"],
        action_evidence_path=flatten_root / "action-evidence.json",
        handoff_path=flatten_root / "handoff.json",
        deps=deps,
    )
    flatten_mapping = _read_handoff(flatten_root / "handoff.json")
    flatten = folding_msa_flatten_handoff_from_mapping(flatten_mapping)
    (logical, projected) = flatten.projected_members[0]
    second_logical = "a3ms/AFDB_AF-0000000000000003_AF-0000000000000004.a3m"
    widened = FoldingMsaFlattenHandoff(
        phase_run_id=flatten.phase_run_id,
        attempt_id=flatten.attempt_id,
        action_id=flatten.action_id,
        predecessor_digest=None,
        projected_members=((logical, projected), (second_logical, projected)),
        local_location=flatten.local_location,
    )
    (flatten_root / "handoff.json").write_text(
        json.dumps(widened.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["split"]
    split_root.mkdir(parents=True)
    evidence_path = split_root / "action-evidence.json"
    run_execute_action(
        phase_runspec_path=tmp_path / "runspec.json",
        action_id=_ACTION_IDS["split"],
        action_evidence_path=evidence_path,
        handoff_path=split_root / "handoff.json",
        deps=deps,
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    # Two targets x two chains each: the actual basenames repeat per target.
    assert evidence["chain_files"] == ["chain_1.a3m", "chain_2.a3m", "chain_1.a3m", "chain_2.a3m"]
    split = folding_split_handoff_from_mapping(_read_handoff(split_root / "handoff.json"))
    assert len(split.targets) == 2
    for target in split.targets:
        assert [Path(path).name for path in target.chain_files] == ["chain_1.a3m", "chain_2.a3m"]


def test_split_derives_pdb_assembly_target_id_from_member_stem(tmp_path: Path) -> None:
    """AC #19: the split action derives target_id == 'pdb_5snm_assembly_1' from a
    pdb_5snm_assembly_1.a3m member stem."""
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    deps = recording.deps()

    flatten_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["msa-flatten"]
    flatten_root.mkdir(parents=True)
    run_execute_action(
        phase_runspec_path=tmp_path / "runspec.json",
        action_id=_ACTION_IDS["msa-flatten"],
        action_evidence_path=flatten_root / "action-evidence.json",
        handoff_path=flatten_root / "handoff.json",
        deps=deps,
    )
    flatten_mapping = _read_handoff(flatten_root / "handoff.json")
    flatten = folding_msa_flatten_handoff_from_mapping(flatten_mapping)
    (_logical, projected) = flatten.projected_members[0]
    pdb_logical = "a3ms/pdb_5snm_assembly_1.a3m"
    pdb_flatten = FoldingMsaFlattenHandoff(
        phase_run_id=flatten.phase_run_id,
        attempt_id=flatten.attempt_id,
        action_id=flatten.action_id,
        predecessor_digest=None,
        projected_members=((pdb_logical, projected),),
        local_location=flatten.local_location,
    )
    (flatten_root / "handoff.json").write_text(
        json.dumps(pdb_flatten.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["split"]
    split_root.mkdir(parents=True)
    run_execute_action(
        phase_runspec_path=tmp_path / "runspec.json",
        action_id=_ACTION_IDS["split"],
        action_evidence_path=split_root / "action-evidence.json",
        handoff_path=split_root / "handoff.json",
        deps=deps,
    )
    split = folding_split_handoff_from_mapping(_read_handoff(split_root / "handoff.json"))
    assert len(split.targets) == 1
    assert split.targets[0].target.target_id == "pdb_5snm_assembly_1"


def test_production_load_runspec_round_trips(tmp_path: Path) -> None:

    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    runspec_path = tmp_path / "runspec.json"
    runspec_path.write_text(json.dumps(runspec.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    loaded = PRODUCTION_EXECUTOR_DEPS.load_runspec(runspec_path)

    assert isinstance(loaded, FoldingPhaseRunSpec)
    assert loaded.phase_run_id == runspec.phase_run_id
    assert loaded.attempt_id == runspec.attempt_id
    assert loaded.payload.backend == "openfold-cli"
    assert loaded.cluster.backend_assets is not None
    assert loaded.cluster.backend_assets.openfold_model_dir == "/models/openfold"


# --- Per-rank fold dispatch + completion journal tests ---


def _widen_msa_flatten_to_second_member(flatten_root: Path) -> None:
    flatten_mapping = _read_handoff(flatten_root / "handoff.json")
    flatten = folding_msa_flatten_handoff_from_mapping(flatten_mapping)
    (logical, projected) = flatten.projected_members[0]
    widened = FoldingMsaFlattenHandoff(
        phase_run_id=flatten.phase_run_id,
        attempt_id=flatten.attempt_id,
        action_id=flatten.action_id,
        predecessor_digest=None,
        projected_members=((logical, projected), (_SECOND_LOGICAL_PATH, projected)),
        local_location=flatten.local_location,
    )
    (flatten_root / "handoff.json").write_text(
        json.dumps(widened.to_mapping(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_upstream_to_preprocess(
    tmp_path: Path, runspec: FoldingPhaseRunSpec, deps: FoldingExecutorDependencies
) -> None:
    """Run msa-flatten -> split -> preprocess over two targets (scalar, rank 0)."""
    _run_action(tmp_path, runspec, deps, _ACTION_IDS["msa-flatten"])
    _widen_msa_flatten_to_second_member(tmp_path / "attempt" / "actions" / _ACTION_IDS["msa-flatten"])
    _run_action(tmp_path, runspec, deps, _ACTION_IDS["split"])
    _run_action(tmp_path, runspec, deps, _ACTION_IDS["preprocess"])


def test_packed_fold_ranks_write_rank_scoped_outputs_and_journal(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()

    # Scalar upstream actions (msa-flatten -> split -> preprocess) over two targets.
    _run_upstream_to_preprocess(tmp_path, runspec, deps)

    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    rank0_summary = _run_action(tmp_path, runspec, deps, _ACTION_IDS["fold"], rank=0)[1]
    rank1_summary = _run_action(tmp_path, runspec, deps, _ACTION_IDS["fold"], rank=1)[1]

    # Each rank created its own backend instance and folded only its target.
    assert len(recording.colabfold_backends) == 2
    assert recording.colabfold_backends[0].runs[0][0].target_id == _TARGET_ID
    assert recording.colabfold_backends[1].runs[0][0].target_id == _SECOND_TARGET_ID
    assert recording.colabfold_backends[0].runs[0][2] == fold_root / "ranks" / "0" / "outputs" / _TARGET_ID
    assert recording.colabfold_backends[1].runs[0][2] == fold_root / "ranks" / "1" / "outputs" / _SECOND_TARGET_ID

    # No shared fold handoff/evidence.
    assert not (fold_root / "handoff.json").exists()
    assert not (fold_root / "action-evidence.json").exists()

    rank0_journal_path = fold_root / "ranks" / "0" / "journal.jsonl"
    rank1_journal_path = fold_root / "ranks" / "1" / "journal.jsonl"
    assert rank0_summary == {"rank": 0, "rank_journal": str(rank0_journal_path), "folded_targets": [_TARGET_ID]}
    assert rank1_summary == {"rank": 1, "rank_journal": str(rank1_journal_path), "folded_targets": [_SECOND_TARGET_ID]}

    rank0_events = read_rank_journal(rank0_journal_path)
    rank1_events = read_rank_journal(rank1_journal_path)
    assert len(rank0_events) == 1
    assert len(rank1_events) == 1

    preprocess_handoff = folding_preprocess_handoff_from_mapping(
        _read_handoff(tmp_path / "attempt" / "actions" / _ACTION_IDS["preprocess"] / "handoff.json")
    )
    fold_action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    expected_qualification = folding_qualification_tuple_id(
        backend="colabfold",
        kernel_image="/images/kernel.sqsh",
        cluster_snapshot_digest=canonical_mapping_digest(runspec.cluster.to_mapping()),
    )
    for event, rank in ((rank0_events[0], 0), (rank1_events[0], 1)):
        expected_target_id = _TARGET_ID if rank == 0 else _SECOND_TARGET_ID
        expected_target = next(
            target for target in preprocess_handoff.targets if target.target.target_id == expected_target_id
        )
        assert event.phase_run_id == runspec.phase_run_id
        assert event.attempt_id == runspec.attempt_id
        assert event.rank == rank
        assert event.fold_action_id == fold_action.action_id
        assert event.fold_action_digest == canonical_mapping_digest(fold_action.to_mapping())
        assert event.shard_projection_sha256 == binding.sha256
        assert event.shard_projection_worker_count == 2
        assert event.shard_projection_lpt_version == 1
        assert event.predecessor_digest == canonical_mapping_digest(preprocess_handoff.to_mapping())
        assert event.target_id == expected_target.target.target_id
        assert event.sequence_sha256 == expected_target.target.sequence_sha256
        assert event.description == expected_target.target.description
        assert event.backend == "colabfold"
        assert event.qualification_tuple_id == expected_qualification
        assert len(event.outputs) == 2
        for output in event.outputs:
            raw = Path(output.path).read_bytes()
            assert output.size == len(raw)
            assert output.sha256 == _sha256(raw)


@pytest.mark.parametrize("nodes,tasks_per_node", [(1, 2), (2, 1)])
def test_packed_fold_topology_shapes_reconcile_worker_count(
    tmp_path: Path,
    nodes: int,
    tasks_per_node: int,
) -> None:
    """Both nontrivial worker shapes reconcile the same contract-owned predicate."""
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    worker_count = nodes * tasks_per_node
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=worker_count,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
        nodes=nodes,
        tasks_per_node=tasks_per_node,
    )
    fold_action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    assert fold_action.resources.is_packed
    assert fold_action.resources.workers == worker_count
    assert fold_action.resources.nodes == nodes
    assert fold_action.resources.tasks_per_node == tasks_per_node
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    assert binding.worker_count == worker_count

    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_packed_fold_ranks(tmp_path, runspec, deps)

    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    assert (fold_root / "ranks" / "0" / "journal.jsonl").exists()
    assert (fold_root / "ranks" / "1" / "journal.jsonl").exists()
    assert not (fold_root / "handoff.json").exists()


def test_packed_fold_binding_worker_count_mismatch_fails_before_dispatch(tmp_path: Path) -> None:
    """A binding whose worker count differs from the fold topology fails closed."""
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    patched_binding = FoldShardProjectionBinding(
        location=binding.location,
        sha256=binding.sha256,
        size_bytes=binding.size_bytes,
        worker_count=3,
        lpt_version=binding.lpt_version,
    )
    runspec = _with_fold_shard_binding(runspec, patched_binding)

    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_upstream_to_preprocess(tmp_path, runspec, deps)

    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    with pytest.raises(FoldingExecutorError, match="worker_count does not match the fold action topology"):
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["fold"],
            action_evidence_path=fold_root / "action-evidence.json",
            handoff_path=fold_root / "handoff.json",
            deps=deps,
            rank=0,
        )
    assert not (fold_root / "ranks").exists()
    assert recording.colabfold_backends == []


def test_openfold_trt_with_projection_fails_closed_without_rank_outputs(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="openfold-trt",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="openfold-trt")
    deps = recording.deps()

    _run_upstream_to_preprocess(tmp_path, runspec, deps)

    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    with pytest.raises(FoldingExecutorError) as excinfo:
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["fold"],
            action_evidence_path=fold_root / "action-evidence.json",
            handoff_path=fold_root / "handoff.json",
            deps=deps,
            rank=0,
        )
    assert str(excinfo.value) == OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR
    assert not (fold_root / "ranks").exists()
    assert not (fold_root / "handoff.json").exists()
    assert not (fold_root / "action-evidence.json").exists()


def test_packed_fold_out_of_range_rank_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_upstream_to_preprocess(tmp_path, runspec, deps)
    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    with pytest.raises(FoldingExecutorError, match="out-of-range rank"):
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["fold"],
            action_evidence_path=fold_root / "action-evidence.json",
            handoff_path=fold_root / "handoff.json",
            deps=deps,
            rank=2,
        )
    assert not (fold_root / "ranks").exists()


def test_packed_fold_projection_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    # Corrupt the projection document bytes without updating the binding digest.
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    projection_path = _staged_projection_path(tmp_path)
    projection_path.write_bytes(b"corrupted")

    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_upstream_to_preprocess(tmp_path, runspec, deps)
    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    with pytest.raises(FoldingExecutorError, match="SHA-256 mismatch"):
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["fold"],
            action_evidence_path=fold_root / "action-evidence.json",
            handoff_path=fold_root / "handoff.json",
            deps=deps,
            rank=0,
        )


def test_packed_fold_worker_count_mismatch_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    # Rewrite the projection document with a different worker_count, then fix the
    # binding digest/size so only the worker_count check fails.
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    projection = FoldShardProjection(
        worker_count=3,
        lpt_version=1,
        ranks=tuple(
            FoldShardRank(
                global_rank=rank,
                targets=(FoldShardTarget(target_id=_TARGET_ID, member_length=100),) if rank == 0 else (),
            )
            for rank in range(3)
        ),
    )
    document = fold_shard_projection_document_bytes(projection)
    projection_path = _staged_projection_path(tmp_path)
    projection_path.write_bytes(document)
    patched_binding = FoldShardProjectionBinding(
        location=binding.location,
        sha256=_sha256(document),
        size_bytes=len(document),
        worker_count=2,
        lpt_version=1,
    )
    runspec = FoldingPhaseRunSpec(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        phase_plan_digest=runspec.phase_plan_digest,
        materialized_at=runspec.materialized_at,
        input_location=runspec.input_location,
        cluster=runspec.cluster,
        payload=FoldingPhaseRunSpecPayload(
            msa_set=runspec.payload.msa_set,
            backend=runspec.payload.backend,
            actions=runspec.payload.actions,
            msa_set_manifest=runspec.payload.msa_set_manifest,
            fold_shard_projection=patched_binding,
        ),
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_upstream_to_preprocess(tmp_path, runspec, deps)
    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    with pytest.raises(FoldingExecutorError, match="worker_count does not match"):
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["fold"],
            action_evidence_path=fold_root / "action-evidence.json",
            handoff_path=fold_root / "handoff.json",
            deps=deps,
            rank=0,
        )


def test_packed_fold_size_mismatch_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    # Keep the digest correct but declare a wrong byte size on the binding.
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    document = _staged_projection_path(tmp_path).read_bytes()
    patched_binding = FoldShardProjectionBinding(
        location=binding.location,
        sha256=binding.sha256,
        size_bytes=len(document) + 1,
        worker_count=binding.worker_count,
        lpt_version=binding.lpt_version,
    )
    runspec = _with_fold_shard_binding(runspec, patched_binding)

    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_upstream_to_preprocess(tmp_path, runspec, deps)
    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    with pytest.raises(FoldingExecutorError, match="size mismatch"):
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["fold"],
            action_evidence_path=fold_root / "action-evidence.json",
            handoff_path=fold_root / "handoff.json",
            deps=deps,
            rank=0,
        )


def test_packed_fold_lpt_version_mismatch_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    # Keep digest/size/worker_count correct but declare a wrong LPT version.
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    patched_binding = FoldShardProjectionBinding(
        location=binding.location,
        sha256=binding.sha256,
        size_bytes=binding.size_bytes,
        worker_count=binding.worker_count,
        lpt_version=binding.lpt_version + 1,
    )
    runspec = _with_fold_shard_binding(runspec, patched_binding)

    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_upstream_to_preprocess(tmp_path, runspec, deps)
    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    with pytest.raises(FoldingExecutorError, match="lpt_version does not match"):
        run_execute_action(
            phase_runspec_path=tmp_path / "runspec.json",
            action_id=_ACTION_IDS["fold"],
            action_evidence_path=fold_root / "action-evidence.json",
            handoff_path=fold_root / "handoff.json",
            deps=deps,
            rank=0,
        )


def test_scalar_binding_worker_count_one_preserves_scalar_layout(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=1,
        rank_target_ids={0: (_TARGET_ID, _SECOND_TARGET_ID)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_upstream_to_preprocess(tmp_path, runspec, deps)

    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    _run_action(tmp_path, runspec, deps, _ACTION_IDS["fold"], rank=0)

    # One backend instance folded both targets into the scalar output layout.
    assert len(recording.colabfold_backends) == 1
    assert [run[0].target_id for run in recording.colabfold_backends[0].runs] == [_TARGET_ID, _SECOND_TARGET_ID]
    assert recording.colabfold_backends[0].runs[0][2] == fold_root / "outputs" / _TARGET_ID
    assert recording.colabfold_backends[0].runs[1][2] == fold_root / "outputs" / _SECOND_TARGET_ID

    # Scalar path publishes the shared fold handoff/evidence, not a rank journal.
    assert (fold_root / "handoff.json").exists()
    assert (fold_root / "action-evidence.json").exists()
    assert not (fold_root / "ranks").exists()
    fold = folding_fold_handoff_from_mapping(_read_handoff(fold_root / "handoff.json"))
    assert [target.target.target_id for target in fold.targets] == [_TARGET_ID, _SECOND_TARGET_ID]


def test_scalar_fold_does_not_read_projection(tmp_path: Path) -> None:
    """A scalar fold action must not require or read the staged projection.

    The binding may still be present (scalar materialization binds a single-rank
    projection), but scalar dispatch selects its layout from the fold action's
    contract-owned packed predicate and never opens the sidecar.
    """
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=1,
        rank_target_ids={0: (_TARGET_ID, _SECOND_TARGET_ID)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    _run_upstream_to_preprocess(tmp_path, runspec, deps)

    # Remove the staged projection sidecar entirely; scalar fold must still run.
    _staged_projection_path(tmp_path).unlink()

    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    _run_action(tmp_path, runspec, deps, _ACTION_IDS["fold"], rank=0)

    assert len(recording.colabfold_backends) == 1
    assert [run[0].target_id for run in recording.colabfold_backends[0].runs] == [_TARGET_ID, _SECOND_TARGET_ID]
    assert (fold_root / "handoff.json").exists()
    assert (fold_root / "action-evidence.json").exists()
    assert not (fold_root / "ranks").exists()


# --- Packed canonical-pair reduce + aggregate fold evidence tests ---


def _run_packed_fold_ranks(
    tmp_path: Path,
    runspec: FoldingPhaseRunSpec,
    deps: FoldingExecutorDependencies,
) -> Path:
    """Run the scalar upstream actions then every packed fold rank."""
    _run_upstream_to_preprocess(tmp_path, runspec, deps)
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    for rank in range(binding.worker_count):
        _run_action(tmp_path, runspec, deps, _ACTION_IDS["fold"], rank=rank)
    return tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]


def _rewrite_rank_journal(path: Path, lines: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8")


def test_packed_reduce_emits_aggregate_and_canonical(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    fold_root = _run_packed_fold_ranks(tmp_path, runspec, deps)
    canonical_root, _ = _run_action(tmp_path, runspec, deps, _ACTION_IDS["canonical-pair"])

    fold_handoff = folding_fold_handoff_from_mapping(_read_handoff(fold_root / "handoff.json"))
    preprocess_handoff = folding_preprocess_handoff_from_mapping(
        _read_handoff(tmp_path / "attempt" / "actions" / _ACTION_IDS["preprocess"] / "handoff.json")
    )
    assert fold_handoff.action_id == _ACTION_IDS["fold"]
    assert fold_handoff.predecessor_digest == canonical_mapping_digest(preprocess_handoff.to_mapping())
    assert [target.target.target_id for target in fold_handoff.targets] == [_TARGET_ID, _SECOND_TARGET_ID]
    assert all(target.pair.tool_used == _COLABFOLD_TOOL_USED for target in fold_handoff.targets)

    aggregate = json.loads((fold_root / "action-evidence.json").read_text(encoding="utf-8"))
    assert [pair["model_entity_id"] for pair in aggregate["pairs"]] == [
        normalize_model_entity_id(_TARGET_ID),
        normalize_model_entity_id(_SECOND_TARGET_ID),
    ]
    assert [pair["tool_used"] for pair in aggregate["pairs"]] == [_COLABFOLD_TOOL_USED, _COLABFOLD_TOOL_USED]

    # The aggregate pairs bind the exact rank-journal outputs.
    for rank, expected_target_id in ((0, _TARGET_ID), (1, _SECOND_TARGET_ID)):
        events = read_rank_journal(fold_root / "ranks" / str(rank) / "journal.jsonl")
        assert len(events) == 1
        event = events[0]
        pair = next(
            pair
            for pair in aggregate["pairs"]
            if pair["model_entity_id"] == normalize_model_entity_id(expected_target_id)
        )
        assert pair["structure_path"] == event.outputs[0].path
        assert pair["scores_path"] == event.outputs[1].path

    # Canonical-pair evidence is accepted and binds the reduced fold handoff.
    canonical = folding_canonical_pair_handoff_from_mapping(_read_handoff(canonical_root / "handoff.json"))
    assert canonical.predecessor_digest == canonical_mapping_digest(fold_handoff.to_mapping())
    combined = json.loads((canonical_root / "action-evidence.json").read_text(encoding="utf-8"))
    validated_index = validate_folding_action_evidence(phase_runspec=runspec, evidence=combined, index_path=None)
    assert validated_index is not None
    assert len(validated_index.entries) == 2
    assert {entry.target_id for entry in validated_index.entries} == {_TARGET_ID, _SECOND_TARGET_ID}


def test_packed_reduce_missing_target_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    fold_root = _run_packed_fold_ranks(tmp_path, runspec, deps)
    (fold_root / "ranks" / "1" / "journal.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(FoldingExecutorError, match="missing fold journal events"):
        _run_action(tmp_path, runspec, deps, _ACTION_IDS["canonical-pair"])


def test_packed_reduce_duplicate_target_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    fold_root = _run_packed_fold_ranks(tmp_path, runspec, deps)
    rank0_journal = fold_root / "ranks" / "0" / "journal.jsonl"
    line = read_rank_journal(rank0_journal)[0].to_mapping()
    _rewrite_rank_journal(rank0_journal, [line, line])

    with pytest.raises(FoldingExecutorError, match="duplicate fold journal event"):
        _run_action(tmp_path, runspec, deps, _ACTION_IDS["canonical-pair"])


def test_packed_reduce_foreign_target_fails_closed(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    fold_root = _run_packed_fold_ranks(tmp_path, runspec, deps)
    rank0_journal = fold_root / "ranks" / "0" / "journal.jsonl"
    foreign = dict(read_rank_journal(rank0_journal)[0].to_mapping())
    foreign["target_id"] = "AF-9999999999999999_AF-9999999999999999"
    _rewrite_rank_journal(rank0_journal, [foreign])

    with pytest.raises(FoldingExecutorError, match="foreign target"):
        _run_action(tmp_path, runspec, deps, _ACTION_IDS["canonical-pair"])


def _mutate_shard_digest(mapping: dict[str, object]) -> None:
    mapping["shard_projection_sha256"] = "0" * 64


def _mutate_rank(mapping: dict[str, object]) -> None:
    mapping["rank"] = 1


def _mutate_sequence_sha256(mapping: dict[str, object]) -> None:
    mapping["sequence_sha256"] = "f" * 64


def _mutate_output_sha256(mapping: dict[str, object]) -> None:
    outputs = mapping["outputs"]
    assert isinstance(outputs, list)
    first = outputs[0]
    assert isinstance(first, dict)
    first["sha256"] = "0" * 64


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_mutate_shard_digest, "shard_projection_sha256 mismatch"),
        (_mutate_rank, "declaring rank 1"),
        (_mutate_sequence_sha256, "sequence_sha256 mismatch"),
        (_mutate_output_sha256, "output mismatch"),
    ],
)
def test_packed_reduce_mismatched_authority_fails_closed(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    expected: str,
) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    fold_root = _run_packed_fold_ranks(tmp_path, runspec, deps)
    rank0_journal = fold_root / "ranks" / "0" / "journal.jsonl"
    tampered = dict(read_rank_journal(rank0_journal)[0].to_mapping())
    mutate(tampered)
    _rewrite_rank_journal(rank0_journal, [tampered])

    with pytest.raises(FoldingExecutorError, match=expected):
        _run_action(tmp_path, runspec, deps, _ACTION_IDS["canonical-pair"])


def test_packed_reduce_torn_final_append_ignored(tmp_path: Path) -> None:
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="colabfold",
        chain_manifest_csv=chain_manifest_csv,
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    recording = _RecordingDeps(runspec, backend="colabfold")
    deps = recording.deps()
    fold_root = _run_packed_fold_ranks(tmp_path, runspec, deps)
    rank0_journal = fold_root / "ranks" / "0" / "journal.jsonl"
    rank0_journal.write_text(
        rank0_journal.read_text(encoding="utf-8") + '{"schema_version": 1, "phase_run_id": "partial',
        encoding="utf-8",
    )

    canonical_root, _ = _run_action(tmp_path, runspec, deps, _ACTION_IDS["canonical-pair"])
    combined = json.loads((canonical_root / "action-evidence.json").read_text(encoding="utf-8"))
    validated_index = validate_folding_action_evidence(phase_runspec=runspec, evidence=combined, index_path=None)
    assert validated_index is not None
    assert len(validated_index.entries) == 2


# --- Install-mode provenance tests ---


def test_handoff_includes_install_mode_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """When BSPP_ORCHESTRATION_SOURCE=override, the handoff includes it."""
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "override")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "abc123")
    mapping: dict[str, object] = {"schema_version": 1, "action_id": "msa-flatten-000001"}
    result = _with_install_provenance(mapping)
    assert result["install_mode"] == "override"
    assert result["orchestration_source_commit"] == "abc123"


def test_handoff_includes_install_mode_baked(monkeypatch: pytest.MonkeyPatch) -> None:
    """When BSPP_ORCHESTRATION_SOURCE=baked and no provenance, only install_mode is set."""
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.delenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", raising=False)
    mapping: dict[str, object] = {"schema_version": 1, "action_id": "msa-flatten-000001"}
    result = _with_install_provenance(mapping)
    assert result["install_mode"] == "baked"
    assert "orchestration_source_commit" not in result


def test_handoff_empty_provenance_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty provenance commit is never injected (truthiness guard)."""
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "baked")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "")
    mapping: dict[str, object] = {"schema_version": 1, "action_id": "msa-flatten-000001"}
    result = _with_install_provenance(mapping)
    assert result["install_mode"] == "baked"
    assert "orchestration_source_commit" not in result


def test_handoff_invalid_install_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid BSPP_ORCHESTRATION_SOURCE raises FoldingExecutorError."""
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "invalid")
    mapping: dict[str, object] = {"schema_version": 1, "action_id": "msa-flatten-000001"}
    with pytest.raises(FoldingExecutorError, match="invalid BSPP_ORCHESTRATION_SOURCE"):
        _with_install_provenance(mapping)


def test_handoff_without_provenance_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A handoff without install_mode/orchestration_source_commit loads with None defaults."""
    monkeypatch.delenv("BSPP_ORCHESTRATION_SOURCE", raising=False)
    monkeypatch.delenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", raising=False)
    mapping: dict[str, object] = {"schema_version": 1, "action_id": "msa-flatten-000001"}
    result = _with_install_provenance(mapping)
    assert "install_mode" not in result
    assert "orchestration_source_commit" not in result


def test_full_chain_empty_provenance_round_trip(tmp_path: Path) -> None:
    """A msa-flatten handoff with no provenance fields loads as a predecessor for split."""
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    outcomes, _attempt_root, error = _run_graph(tmp_path, runspec, recording.deps())
    assert error is None
    msa_flatten_mapping = _read_handoff(outcomes[_ACTION_IDS["msa-flatten"]]["handoff_path"])
    assert "install_mode" not in msa_flatten_mapping
    assert "orchestration_source_commit" not in msa_flatten_mapping
    # The split action loads the msa-flatten handoff as its predecessor
    split_mapping = _read_handoff(outcomes[_ACTION_IDS["split"]]["handoff_path"])
    assert "predecessor_digest" in split_mapping


def test_full_chain_with_provenance_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A msa-flatten handoff WITH provenance fields loads as a predecessor for split,
    and the split action's predecessor digest is stable across re-reads."""
    monkeypatch.setenv("BSPP_ORCHESTRATION_SOURCE", "override")
    monkeypatch.setenv("BSPP_ORCHESTRATION_PROVENANCE_COMMIT", "abc123def456")
    chain_manifest_csv = (_FIXTURE_DIR / "chain-manifest.csv").resolve()
    runspec, _ = _make_runspec(tmp_path, backend="openfold-cli", chain_manifest_csv=chain_manifest_csv)
    recording = _RecordingDeps(runspec, backend="openfold-cli")
    outcomes, _attempt_root, error = _run_graph(tmp_path, runspec, recording.deps())
    assert error is None
    msa_flatten_mapping = _read_handoff(outcomes[_ACTION_IDS["msa-flatten"]]["handoff_path"])
    assert msa_flatten_mapping["install_mode"] == "override"
    assert msa_flatten_mapping["orchestration_source_commit"] == "abc123def456"
    # The split action loads the msa-flatten handoff as its predecessor
    split_mapping = _read_handoff(outcomes[_ACTION_IDS["split"]]["handoff_path"])
    assert "predecessor_digest" in split_mapping
    # Verify the predecessor digest is stable: re-read the msa-flatten handoff
    msa_flatten_loaded = folding_msa_flatten_handoff_from_mapping(msa_flatten_mapping)
    digest1 = canonical_mapping_digest(msa_flatten_loaded.to_mapping())
    # Re-read from disk and compute again
    msa_flatten_reread = folding_msa_flatten_handoff_from_mapping(
        _read_handoff(outcomes[_ACTION_IDS["msa-flatten"]]["handoff_path"])
    )
    digest2 = canonical_mapping_digest(msa_flatten_reread.to_mapping())
    assert digest1 == digest2


def test_mixed_bioir_policy_survives_packed_journals_and_canonical_reduction(tmp_path: Path) -> None:
    """The real split/preprocess/journal/reducer path preserves selected model science."""
    from dataclasses import replace

    from bspp.orchestration.contract.folding_bioir import BioIRModelPolicy

    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="bioir",
        chain_manifest_csv=(_FIXTURE_DIR / "chain-manifest.csv").resolve(),
        worker_count=2,
        rank_target_ids={0: (_TARGET_ID,), 1: (_SECOND_TARGET_ID,)},
    )
    mono_checkpoint = tmp_path / "monomer.pt"
    multi_checkpoint = tmp_path / "multimer.pt"
    mono_checkpoint.write_bytes(b"monomer diagnostic")
    multi_checkpoint.write_bytes(b"multimer diagnostic")
    policy = BioIRModelPolicy(
        monomer_checkpoint_sha256=_sha256(mono_checkpoint.read_bytes()),
        monomer_checkpoint_size_bytes=mono_checkpoint.stat().st_size,
        multimer_checkpoint_sha256=_sha256(multi_checkpoint.read_bytes()),
        multimer_checkpoint_size_bytes=multi_checkpoint.stat().st_size,
    )
    actions = tuple(
        replace(
            action,
            payload=replace(
                action.payload, params=(*action.payload.params, ("bioir_model_policy_digest", policy.digest))
            ),
        )
        if action.action_kind in {"fold", "canonical-pair"}
        else action
        for action in runspec.payload.actions
    )
    runspec = replace(
        runspec,
        payload=replace(runspec.payload, actions=actions, bioir_model_policy=policy),
        cluster=replace(
            runspec.cluster,
            backend_assets=FoldingBackendAssetsSnapshot(
                backend="bioir",
                bioir_checkpoint=str(multi_checkpoint),
                bioir_monomer_checkpoint=str(mono_checkpoint),
            ),
        ),
    )
    sessions = []

    class PolicySession(RecordingBioIRSession):
        def __init__(self, checkpoint, output_dir, *, model_source):
            super().__init__(checkpoint, output_dir)
            self.model_source = model_source
            sessions.append(self)

        def run(self, target, prepared, output_dir):
            result = super().run(target, prepared, output_dir)
            tool = "OpenFold2 (BioNeMo IR) / " + (
                "OpenFold-pTM" if self.model_source == "openfold2_ptm_1" else "AlphaFold-Multimer"
            )
            size = sum(map(len, target.chains))
            result.predictions[0].scores_path.write_text(
                serialize_scores_json(
                    plddt=[80.0] * size,
                    pae=[[0.1] * size for _ in range(size)],
                    max_pae=0.1,
                    ptm=0.8,
                    iptm=None if len(target.chains) == 1 else 0.9,
                )
            )
            return replace(result, metadata={"tool_used": tool, "model_source": self.model_source})

    deps = replace(_RecordingDeps(runspec, "bioir").deps(), bioir_session_factory=PolicySession)
    flatten_root, _ = _run_action(tmp_path, runspec, deps, _ACTION_IDS["msa-flatten"])
    flatten = folding_msa_flatten_handoff_from_mapping(_read_handoff(flatten_root / "handoff.json"))
    monomer_a3m = tmp_path / "synthetic-monomer.a3m"
    homomer_a3m = tmp_path / "synthetic-homomer.a3m"
    monomer_a3m.write_text("#2\t1\n>query\nAA\n>hit\nAC\n")
    homomer_a3m.write_text("#2\t2\n>query\nAA\n>hit\nAC\n")
    # Test-only input construction, before the actual split and preprocessing.
    flatten = replace(
        flatten, projected_members=((_LOGICAL_PATH, str(monomer_a3m)), (_SECOND_LOGICAL_PATH, str(homomer_a3m)))
    )
    (flatten_root / "handoff.json").write_text(json.dumps(flatten.to_mapping()))
    _run_action(tmp_path, runspec, deps, _ACTION_IDS["split"])
    _run_action(tmp_path, runspec, deps, _ACTION_IDS["preprocess"])
    for rank in range(2):
        _run_action(tmp_path, runspec, deps, _ACTION_IDS["fold"], rank=rank)
    canonical_root, _ = _run_action(tmp_path, runspec, deps, _ACTION_IDS["canonical-pair"])
    fold_root = tmp_path / "attempt" / "actions" / _ACTION_IDS["fold"]
    fold = folding_fold_handoff_from_mapping(_read_handoff(fold_root / "handoff.json"))
    assert [len(target.target.chains) for target in fold.targets] == [1, 2]
    assert fold.targets[1].target.chains == ("AA", "AA")
    assert [session.checkpoint for session in sessions] == [mono_checkpoint, multi_checkpoint]
    assert all(session.closed for session in sessions)
    assert [target.model_metadata["model_source"] for target in fold.targets] == [
        "openfold2_ptm_1",
        "alphafold2_multimer_1",
    ]
    assert [target.model_metadata["checkpoint_sha256"] for target in fold.targets] == [
        policy.monomer_checkpoint_sha256,
        policy.multimer_checkpoint_sha256,
    ]
    assert all(target.model_metadata["bioir_model_policy_digest"] == policy.digest for target in fold.targets)
    assert fold.targets[0].pair.scores.iptm is None
    assert len(fold.targets[0].pair.scores.pae) == 2
    combined = _read_handoff(canonical_root / "action-evidence.json")
    index = validate_folding_action_evidence(phase_runspec=runspec, evidence=combined, index_path=None)
    assert index is not None
    assert [entry.tool_used for entry in index.entries] == [
        "OpenFold2 (BioNeMo IR) / OpenFold-pTM",
        "OpenFold2 (BioNeMo IR) / AlphaFold-Multimer",
    ]
