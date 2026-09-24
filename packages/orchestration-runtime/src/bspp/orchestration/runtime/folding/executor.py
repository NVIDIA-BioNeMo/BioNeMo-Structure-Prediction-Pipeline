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

"""Job-local folding Runtime Action executor.

This module is the executable entry point each rendered folding action invokes
inside its job-local Runtime container. It parses the quoted argv, strict-loads
one :class:`~bspp.orchestration.contract.phase.FoldingPhaseRunSpec` through the
family loader (rejecting preprocessing/postprocessing authorities), finds
exactly one matching action, validates the handoff/evidence/output path
identity, and dispatches to one of the five immutable action handlers:

``msa-flatten -> split -> preprocess -> fold -> canonical-pair``

Every handoff/evidence record is published atomically (temp file + fsync +
``os.replace``) with strict identity binding: outputs are produced
first, the handoff is published second, and the action evidence is published
last. That one-atomic-handoff rule holds for every scalar action, but the
packed fold action's handoff is the exact rank-owned journal set, and the
canonical-pair action publishes the sole aggregate fold evidence/handoff.
The terminal ``canonical-pair`` action validates and aggregates all
five per-action evidence records into the finalization-ready combined mapping
and binds the written canonical index through its digest.

Backend/transport seams are injectable via :class:`FoldingExecutorDependencies`;
production factories import the scientific backends lazily and translate an
``ImportError`` into the precise preflight error. ``openfold-trt``
fails closed before any publication with the deferred ``model_fn`` factory
error.

Target identity decision recorded here: the
``FoldingTargetIdentity.description`` is the declared member logical path, which
is non-empty, manifest-derived, and deterministic.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from bspp.orchestration.contract.folding_artifact_evidence import ARTIFACT_EVIDENCE_PROFILE, MAX_SCORE_BYTES
from bspp.orchestration.contract.folding_carry_forward import (
    FoldingCarryForwardRecord,
    folding_carry_forward_record_from_mapping,
)
from bspp.orchestration.contract.folding_evidence import (
    CanonicalPairActionEvidence,
    FoldActionEvidence,
    MsaFlattenActionEvidence,
    PreprocessActionEvidence,
    SplitActionEvidence,
)
from bspp.orchestration.contract.folding_execution import (
    OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR,
    FoldingCanonicalPairEntry,
    FoldingCanonicalPairHandoff,
    FoldingFoldHandoff,
    FoldingFoldTarget,
    FoldingMsaFlattenHandoff,
    FoldingPreprocessHandoff,
    FoldingPreprocessTarget,
    FoldingSplitHandoff,
    FoldingSplitTarget,
    FoldingTargetIdentity,
    folding_action_kind_for_action_id,
    folding_handoff_from_mapping,
    folding_target_sequence_sha256,
)
from bspp.orchestration.contract.folding_shard import (
    FoldShardProjection,
    fold_shard_projection_from_mapping,
    fold_shard_projection_staged_basename,
)
from bspp.orchestration.contract.model_identity import normalize_model_entity_id
from bspp.orchestration.contract.phase import (
    FoldingPhaseRunSpec,
    FoldingRuntimeAction,
    canonical_mapping_digest,
    phase_runspec_family_from_mapping,
)
from bspp.orchestration.contract.prediction_pair import (
    PredictionPair,
    prediction_scores_payload_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import (
    MsaArtifactSetManifest,
    VerifiedLocalBundledArtifactLocation,
)
from bspp.orchestration.contract.runspec import VALID_TOOL_USED

from .carry_adoption import (
    CarryAdoptionAuthorityBinding,
    adopt_all_carried_outputs,
    adopt_carried_outputs,
    read_adopted_journal,
)
from .execution.a3m_split import _a3m_query_chains, split_merged_a3m
from .execution.bioir_inputs import BioIRInputPreprocessor
from .execution.bioir_policy import BioIRPolicySessions, bioir_model_metadata
from .execution.chain_manifest import ChainManifest, classify_target, parse_chain_manifest
from .execution.colabfold_backend import ColabFoldConfig
from .execution.colabfold_inputs import prepare as _colabfold_prepare
from .execution.errors import FoldingBackendError
from .execution.models import FoldingResult, PreparedInput, ProteinTarget
from .execution.msa_intake import prepare_folding_msa_input
from .execution.msa_models import MSAResult
from .execution.msa_preparation import msa_result_from_split_paths
from .execution.openfold_inputs import OpenFoldInputPreprocessor
from .rank_journal import (
    RankJournalEvent,
    RankJournalOutput,
    RankJournalWriter,
    folding_qualification_tuple_id,
    read_rank_journal,
)

if TYPE_CHECKING:
    from .execution.bioir_session import BioIRFoldSession
    from .execution.colabfold_backend import ColabFoldBackend
    from .execution.openfold_cli import OpenFoldCliBackend

# the pinned LZ4 decompression argv used for remote MSA bundles.
_PINNED_LZ4_ARGV = ("/usr/bin/lz4", "-d", "-c")

_PREFLIGHT_ERROR_TEMPLATE = (
    "backend image preflight failed: required BSPP Runtime/Contract interface unavailable ({module})"
)


class FoldingExecutorError(RuntimeError):
    """One folding Runtime Action failed closed."""


PredecessorHandoff = (
    FoldingMsaFlattenHandoff
    | FoldingSplitHandoff
    | FoldingPreprocessHandoff
    | FoldingFoldHandoff
    | FoldingCanonicalPairHandoff
)


@dataclass(frozen=True)
class ShardSelection:
    """One verified canonical shard projection and the rank's ordered target ids."""

    projection: FoldShardProjection
    target_ids: tuple[str, ...]
    sha256: str
    worker_count: int
    lpt_version: int


@dataclass(frozen=True)
class FoldingExecutorDependencies:
    """One seam per external boundary; production defaults live in ``PRODUCTION_EXECUTOR_DEPS``."""

    load_runspec: Callable[[Path], FoldingPhaseRunSpec]
    msa_intake: Callable[..., tuple[VerifiedLocalBundledArtifactLocation, dict[str, Path]]]
    split_merged: Callable[[Path, ProteinTarget, Path], list[Path]]
    openfold_preprocess: Callable[[], OpenFoldInputPreprocessor]
    bioir_preprocess: Callable[[], BioIRInputPreprocessor]
    colabfold_prepare: Callable[[ProteinTarget, MSAResult, Path], PreparedInput]
    openfold_backend_factory: Callable[[Path, ChainManifest], OpenFoldCliBackend]
    colabfold_backend_factory: Callable[[ColabFoldConfig, ChainManifest], ColabFoldBackend]
    bioir_session_factory: Callable[..., BioIRFoldSession]
    parse_chain_manifest: Callable[[Path], ChainManifest]


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            # Stream the existing exact format: cohort evidence includes full
            # PAE matrices and must not build a second giant serialized buffer.
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FoldingExecutorError(f"cannot read {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FoldingExecutorError(f"malformed JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FoldingExecutorError(f"expected a JSON object in {path}")
    return payload


def _fold_action_param(action: FoldingRuntimeAction, key: str) -> str:
    """Return one exact fold action param value or fail closed."""
    for param_key, value in action.payload.params:
        if param_key == key:
            return value
    raise FoldingExecutorError(f"fold action is missing its {key} param")


def _fold_shard_projection_path(phase_runspec_path: Path, location: str) -> Path:
    """Resolve the canonical projection beside the staged Phase RunSpec.

    The binding ``location`` is the control-side authority-relative path
    (``attempts/<attempt>/fold-shard-projection.json``). The cluster does not
    replicate the control authority tree; it stages the projection document
    beside the immutable RunSpec it already mounts, so the executor resolves the
    canonical staged basename against the RunSpec's own directory rather than
    re-deriving either the control-side authority layout or
    ``cluster.staging_root``.
    """
    return phase_runspec_path.parent / fold_shard_projection_staged_basename(location)


def _parse_json_bytes(raw: bytes, path: Path) -> Mapping[str, object]:
    """Strict-load a JSON object from already-read bytes (no second read)."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FoldingExecutorError(f"malformed JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FoldingExecutorError(f"expected a JSON object in {path}")
    return payload


def _load_and_verify_shard_projection(
    phase_runspec_path: Path,
    runspec: FoldingPhaseRunSpec,
    rank: int,
) -> ShardSelection | None:
    """Load, verify, and select one rank's row from the canonical shard projection.

    ``None`` means the RunSpec carries no shard binding (legacy scalar authority).
    The document is resolved beside the staged RunSpec (see
    :func:`_fold_shard_projection_path`). Its bytes are read once, digest/size
    verified, and parsed from that same byte string (no TOCTOU re-open). Any
    digest/size/worker-count/LPT-version mismatch, or an out-of-range rank, fails
    closed before any fold dispatch.
    """
    binding = runspec.payload.fold_shard_projection
    if binding is None:
        return None
    location = _fold_shard_projection_path(phase_runspec_path, binding.location)
    try:
        raw = location.read_bytes()
    except OSError as exc:
        raise FoldingExecutorError(f"cannot read fold shard projection {location}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != binding.sha256:
        raise FoldingExecutorError(f"fold shard projection SHA-256 mismatch for {location}")
    if len(raw) != binding.size_bytes:
        raise FoldingExecutorError(f"fold shard projection size mismatch for {location}")
    projection = fold_shard_projection_from_mapping(_parse_json_bytes(raw, location))
    if projection.worker_count != binding.worker_count:
        raise FoldingExecutorError("fold shard projection worker_count does not match the RunSpec binding")
    if projection.lpt_version != binding.lpt_version:
        raise FoldingExecutorError("fold shard projection lpt_version does not match the RunSpec binding")
    if rank < 0 or rank >= projection.worker_count:
        raise FoldingExecutorError(f"out-of-range rank {rank} for worker_count {projection.worker_count}")
    target_ids = tuple(target.target_id for target in projection.ranks[rank].targets)
    return ShardSelection(
        projection=projection,
        target_ids=target_ids,
        sha256=binding.sha256,
        worker_count=projection.worker_count,
        lpt_version=projection.lpt_version,
    )


def _load_packed_shard_selection(
    phase_runspec_path: Path,
    runspec: FoldingPhaseRunSpec,
    fold_action: FoldingRuntimeAction,
    rank: int,
) -> ShardSelection:
    """Load and verify the packed fold shard projection for one rank.

    Packed execution requires a binding whose worker count equals the fold
    action's contract-owned worker count before any rank selection, reduction,
    or adoption. Scalar execution never calls this helper and therefore never
    reads the projection.
    """
    if not fold_action.resources.is_packed:
        raise FoldingExecutorError("packed fold execution requires a packed fold action topology")
    binding = runspec.payload.fold_shard_projection
    if binding is None:
        raise FoldingExecutorError("packed fold execution requires a fold shard projection binding")
    if binding.worker_count != fold_action.resources.workers:
        raise FoldingExecutorError("fold shard projection worker_count does not match the fold action topology")
    selection = _load_and_verify_shard_projection(phase_runspec_path, runspec, rank)
    if selection is None:
        raise FoldingExecutorError("packed fold execution requires a fold shard projection binding")
    return selection


def _file_output(path: Path) -> RankJournalOutput:
    """Return the exact size and streaming SHA-256 of one produced output file."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return RankJournalOutput(path=str(path), size=size, sha256=digest.hexdigest())


def _load_carry_record(path: Path) -> FoldingCarryForwardRecord:
    """Strict-load one sealed folding carry record document."""
    payload = _read_json_mapping(path)
    try:
        return folding_carry_forward_record_from_mapping(payload)
    except ValueError as exc:
        raise FoldingExecutorError(f"invalid folding carry record {path}: {exc}") from exc


def _build_carry_adoption_binding(
    runspec: FoldingPhaseRunSpec,
    fold_action: FoldingRuntimeAction,
    preprocess: FoldingPreprocessHandoff,
    selection: ShardSelection,
) -> CarryAdoptionAuthorityBinding:
    """Build the successor authority tuple for one adopted journal event."""
    return CarryAdoptionAuthorityBinding(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        fold_action_id=fold_action.action_id,
        fold_action_digest=canonical_mapping_digest(fold_action.to_mapping()),
        shard_projection_sha256=selection.sha256,
        shard_projection_worker_count=selection.worker_count,
        shard_projection_lpt_version=selection.lpt_version,
        predecessor_digest=canonical_mapping_digest(preprocess.to_mapping()),
        backend=runspec.payload.backend,
        qualification_tuple_id=folding_qualification_tuple_id(
            backend=runspec.payload.backend,
            kernel_image=_fold_action_param(fold_action, "kernel_image"),
            cluster_snapshot_digest=canonical_mapping_digest(runspec.cluster.to_mapping()),
        ),
        descriptions={target.target.target_id: target.target.description for target in preprocess.targets},
    )


def _production_load_runspec(path: Path) -> FoldingPhaseRunSpec:
    try:
        payload = yaml.safe_load(path.read_bytes())
    except yaml.YAMLError as exc:
        raise FoldingExecutorError(f"Invalid folding Phase RunSpec YAML/JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FoldingExecutorError(f"Expected folding Phase RunSpec mapping in {path}")
    runspec = phase_runspec_family_from_mapping(payload)
    if not isinstance(runspec, FoldingPhaseRunSpec):
        raise FoldingExecutorError(f"Phase RunSpec at {path} is not a folding Phase RunSpec ({type(runspec).__name__})")
    return runspec


def _production_msa_intake(
    runspec: FoldingPhaseRunSpec,
    workspace: Path,
    *,
    artifact_set: MsaArtifactSetManifest,
    lz4_argv: tuple[str, ...],
) -> tuple[VerifiedLocalBundledArtifactLocation, dict[str, Path]]:
    return prepare_folding_msa_input(runspec, workspace, artifact_set=artifact_set, lz4_argv=lz4_argv)


def _production_openfold_preprocess() -> OpenFoldInputPreprocessor:
    return OpenFoldInputPreprocessor()


def _production_bioir_preprocess() -> BioIRInputPreprocessor:
    return BioIRInputPreprocessor()


def _production_openfold_backend_factory(model_dir: Path, manifest: ChainManifest) -> OpenFoldCliBackend:
    try:
        from .execution.openfold_cli import OpenFoldCliBackend
    except ImportError as exc:
        raise FoldingExecutorError(
            _PREFLIGHT_ERROR_TEMPLATE.format(module="bspp.orchestration.runtime.folding.execution.openfold_cli")
        ) from exc
    return OpenFoldCliBackend(model_dir=model_dir)


def _production_colabfold_backend_factory(config: ColabFoldConfig, manifest: ChainManifest) -> ColabFoldBackend:
    try:
        from .execution.colabfold_backend import ColabFoldBackend
    except ImportError as exc:
        raise FoldingExecutorError(
            _PREFLIGHT_ERROR_TEMPLATE.format(module="bspp.orchestration.runtime.folding.execution.colabfold_backend")
        ) from exc
    return ColabFoldBackend(config=config, chain_manifest=manifest)


def _production_bioir_session_factory(
    checkpoint: Path, output_dir: Path, *, model_source: str = "alphafold2_multimer_1"
) -> BioIRFoldSession:
    try:
        from .execution.bioir_config import OpenFoldModelSettings, OpenFoldSettings
        from .execution.bioir_session import BioIRFoldSession
    except ImportError as exc:
        raise FoldingExecutorError(
            _PREFLIGHT_ERROR_TEMPLATE.format(module="bspp.orchestration.runtime.folding.execution.bioir_session")
        ) from exc
    if model_source == "openfold2_ptm_1":
        model = OpenFoldModelSettings(
            model_id="openfold2_ptm_1",
            model_preset="openfold2_ptm_1",
            parameter_file="finetuning_ptm_1.pt",
            seed=0,
            model_source=model_source,
        )
    elif model_source == "alphafold2_multimer_1":
        model = OpenFoldModelSettings(
            model_id="model_1_multimer_v3",
            model_preset="model_1_multimer_v3",
            parameter_file="params_model_1_multimer_v3.pt",
            seed=0,
        )
    else:
        raise FoldingExecutorError(f"unsupported BioIR model policy source: {model_source}")
    settings = OpenFoldSettings()
    return BioIRFoldSession(model=model, settings=settings, checkpoint=checkpoint, output_dir=output_dir)


PRODUCTION_EXECUTOR_DEPS = FoldingExecutorDependencies(
    load_runspec=_production_load_runspec,
    msa_intake=_production_msa_intake,
    split_merged=split_merged_a3m,
    openfold_preprocess=_production_openfold_preprocess,
    bioir_preprocess=_production_bioir_preprocess,
    colabfold_prepare=_colabfold_prepare,
    openfold_backend_factory=_production_openfold_backend_factory,
    colabfold_backend_factory=_production_colabfold_backend_factory,
    bioir_session_factory=_production_bioir_session_factory,
    parse_chain_manifest=parse_chain_manifest,
)


def _validate_handoff_path_identity(handoff_path: Path, action_id: str) -> tuple[Path, Path, Path]:
    resolved = handoff_path.resolve()
    if resolved.name != "handoff.json" or resolved.parent.name != action_id or resolved.parent.parent.name != "actions":
        raise FoldingExecutorError(
            f"handoff path {handoff_path} must be <attempt-root>/actions/{action_id}/handoff.json"
        )
    action_root = resolved.parent
    actions_dir = action_root.parent
    attempt_root = actions_dir.parent
    return action_root, actions_dir, attempt_root


def _validate_evidence_path_confinement(action_evidence_path: Path, action_root: Path) -> None:
    evidence_parent = action_evidence_path.resolve().parent
    if evidence_parent != action_root and action_root not in evidence_parent.parents:
        raise FoldingExecutorError(f"action evidence path {action_evidence_path} escapes the action root {action_root}")


def _reject_partial_rerun_content(action_root: Path, *, rank: int, packed: bool) -> None:
    """Reject any pre-existing action-root content beyond scheduler logs (and, for
    a packed fold action, other ranks' rank-scoped content).

    The rendered sbatch creates the action root and writes only
    ``slurm-<jobid>.out``/``.err`` into it before this process starts. An
    interrupted earlier attempt (outputs written, handoff/evidence never
    published) leaves the root otherwise nonempty; dispatching over that state
    could mix stale partial outputs into a later publication, and the
    dispatchers are not required to clean it consistently. Fail closed instead.

    Packed fold actions run many ranks concurrently under one action root, so a
    rank may legitimately see other ranks' ``ranks/<other>/...`` content already
    present. Only this rank's own ``ranks/<rank>/...`` (stale partial content)
    and anything outside ``ranks/`` (e.g. a stray shared ``handoff.json``) is
    rejected for a packed rank.
    """
    if not action_root.is_dir():
        return

    def is_scheduler_log(path: Path) -> bool:
        return (
            path.is_file()
            and path.parent == action_root
            and path.name.startswith("slurm-")
            and path.suffix in {".out", ".err"}
        )

    ranks_root = action_root / "ranks"

    def is_allowed(path: Path) -> bool:
        if is_scheduler_log(path):
            return True
        if not packed:
            return False
        try:
            relative = path.relative_to(ranks_root)
        except ValueError:
            return False
        if not relative.parts:
            return True  # the ranks/ directory itself
        return relative.parts[0] != str(rank)

    stray = sorted(path for path in action_root.rglob("*") if not is_allowed(path))
    if stray:
        raise FoldingExecutorError(
            f"action root {action_root} is not fresh: {stray[0]} already exists; "
            "cancel the Attempt and retry for a clean successor instead of reusing partial outputs"
        )


def _validate_predecessor_action_ids(action: FoldingRuntimeAction, runspec: FoldingPhaseRunSpec) -> None:
    known = {candidate.action_id for candidate in runspec.payload.actions}
    for dependency in action.dependencies:
        if dependency not in known:
            raise FoldingExecutorError(f"action {action.action_id} has unknown dependency {dependency}")


def _single_predecessor(action: FoldingRuntimeAction) -> str | None:
    dependencies = action.dependencies
    kind = action.action_kind
    if kind == "msa-flatten":
        if dependencies:
            raise FoldingExecutorError("msa-flatten action must have no dependencies")
        return None
    if len(dependencies) != 1:
        raise FoldingExecutorError(f"{kind} action requires exactly one dependency")
    return dependencies[0]


def _load_predecessor_handoff(
    path: Path,
    runspec: FoldingPhaseRunSpec,
    dependency_id: str,
    expected_kind: str,
) -> PredecessorHandoff:
    payload = _read_json_mapping(path)
    handoff = folding_handoff_from_mapping(payload)
    if folding_action_kind_for_action_id(handoff.action_id) != expected_kind:
        raise FoldingExecutorError(f"predecessor handoff {path} is not a {expected_kind} handoff")
    if handoff.phase_run_id != runspec.phase_run_id:
        raise FoldingExecutorError(f"predecessor handoff phase_run_id does not match the RunSpec: {path}")
    if handoff.attempt_id != runspec.attempt_id:
        raise FoldingExecutorError(f"predecessor handoff attempt_id does not match the RunSpec: {path}")
    if handoff.action_id != dependency_id:
        raise FoldingExecutorError(
            f"predecessor handoff action_id {handoff.action_id!r} does not match dependency {dependency_id!r}: {path}"
        )
    return handoff


def _dispatch_msa_flatten(
    runspec: FoldingPhaseRunSpec,
    action: FoldingRuntimeAction,
    action_root: Path,
    deps: FoldingExecutorDependencies,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    manifest = runspec.payload.msa_set_manifest
    if manifest is None:
        raise FoldingExecutorError("msa-flatten cannot execute a historical authority without a MSA set manifest")
    workspace = action_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    local_location, projected = deps.msa_intake(runspec, workspace, artifact_set=manifest, lz4_argv=_PINNED_LZ4_ARGV)
    if set(projected) != set(runspec.payload.msa_set.member_a3m_paths):
        raise FoldingExecutorError("projected MSA members do not match the declared member paths")
    projected_members = tuple(
        (logical, str(projected[logical])) for logical in runspec.payload.msa_set.member_a3m_paths
    )
    handoff = FoldingMsaFlattenHandoff(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        action_id=action.action_id,
        predecessor_digest=None,
        projected_members=projected_members,
        local_location=local_location.to_mapping(),
    )
    evidence: dict[str, object] = {"a3m_paths": [logical for logical, _ in projected_members]}
    return handoff.to_mapping(), evidence


def _dispatch_split(
    runspec: FoldingPhaseRunSpec,
    action: FoldingRuntimeAction,
    action_root: Path,
    actions_dir: Path,
    deps: FoldingExecutorDependencies,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    dependency_id = _single_predecessor(action)
    assert dependency_id is not None
    predecessor = _load_predecessor_handoff(
        actions_dir / dependency_id / "handoff.json", runspec, dependency_id, "msa-flatten"
    )
    if not isinstance(predecessor, FoldingMsaFlattenHandoff):
        raise FoldingExecutorError("split predecessor must be an msa-flatten handoff")
    predecessor_digest = canonical_mapping_digest(predecessor.to_mapping())

    targets: list[FoldingSplitTarget] = []
    for logical_path, projected_path in predecessor.projected_members:
        target_id = normalize_model_entity_id(Path(logical_path).stem)
        chains = _a3m_query_chains(Path(projected_path))
        if not chains:
            raise FoldingExecutorError(f"merged A3M has no query chains: {projected_path}")
        sequence_sha256 = folding_target_sequence_sha256(chains)
        ptarget = ProteinTarget(target_id=target_id, description=logical_path, chains=chains)
        out_dir = action_root / "splits" / target_id
        paths = deps.split_merged(Path(projected_path), ptarget, out_dir)
        if len(paths) != len(chains):
            raise FoldingExecutorError(f"split produced {len(paths)} chains for {len(chains)} target chains")
        targets.append(
            FoldingSplitTarget(
                target=FoldingTargetIdentity(
                    target_id=target_id,
                    description=logical_path,
                    chains=chains,
                    sequence_sha256=sequence_sha256,
                ),
                merged_source=projected_path,
                chain_files=tuple(str(path) for path in paths),
                chain_count=len(paths),
            )
        )

    handoff = FoldingSplitHandoff(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        action_id=action.action_id,
        predecessor_digest=predecessor_digest,
        targets=tuple(targets),
    )
    # Evidence carries the ACTUAL produced filenames (basenames in target then
    # chain order). Each target's split restarts at chain_1.a3m beneath its own
    # directory, so a multi-target run legitimately repeats names here; the
    # disambiguated absolute paths live in the split handoff.
    evidence: dict[str, object] = {
        "chain_files": [Path(path).name for target in targets for path in target.chain_files]
    }
    return handoff.to_mapping(), evidence


def _dispatch_preprocess(
    runspec: FoldingPhaseRunSpec,
    action: FoldingRuntimeAction,
    action_root: Path,
    actions_dir: Path,
    deps: FoldingExecutorDependencies,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    dependency_id = _single_predecessor(action)
    assert dependency_id is not None
    predecessor = _load_predecessor_handoff(
        actions_dir / dependency_id / "handoff.json", runspec, dependency_id, "split"
    )
    if not isinstance(predecessor, FoldingSplitHandoff):
        raise FoldingExecutorError("preprocess predecessor must be a split handoff")
    predecessor_digest = canonical_mapping_digest(predecessor.to_mapping())

    backend = runspec.payload.backend
    if backend == "openfold-cli":
        layout = "openfold"
    elif backend in {"bioir", "openfold-trt"}:
        layout = "bioir"
    elif backend == "colabfold":
        layout = "colabfold"
    else:
        raise FoldingExecutorError(f"unsupported folding backend: {backend!r}")

    targets: list[FoldingPreprocessTarget] = []
    for split_target in predecessor.targets:
        ptarget = ProteinTarget(
            target_id=split_target.target.target_id,
            description=split_target.target.description,
            chains=split_target.target.chains,
        )
        msa = msa_result_from_split_paths(
            [Path(path) for path in split_target.chain_files],
            ptarget,
            artifact_set_id=runspec.payload.msa_set.artifact_set_id,
            selected_logical_path=split_target.target.description,
            merged_source_path=Path(split_target.merged_source),
        )
        out_dir = action_root / "prepared" / split_target.target.target_id
        if backend == "openfold-cli":
            prepared = deps.openfold_preprocess().run(ptarget, msa, out_dir)
        elif backend in {"bioir", "openfold-trt"}:
            prepared = deps.bioir_preprocess().run(ptarget, msa, out_dir)
        elif backend == "colabfold":
            prepared = deps.colabfold_prepare(ptarget, msa, out_dir)
        else:
            raise FoldingExecutorError(f"unsupported folding backend: {backend!r}")
        targets.append(
            FoldingPreprocessTarget(
                target=split_target.target,
                layout=layout,
                fasta_dir=str(prepared.fasta_dir),
                alignment_dir=str(prepared.alignment_dir),
                template_dir=str(prepared.template_dir),
            )
        )

    handoff = FoldingPreprocessHandoff(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        action_id=action.action_id,
        predecessor_digest=predecessor_digest,
        targets=tuple(targets),
    )
    evidence: dict[str, object] = {"fasta_dir": "fasta", "alignment_dir": "alignments", "layout": layout}
    return handoff.to_mapping(), evidence


def _dispatch_fold(
    phase_runspec_path: Path,
    runspec: FoldingPhaseRunSpec,
    action: FoldingRuntimeAction,
    action_root: Path,
    actions_dir: Path,
    deps: FoldingExecutorDependencies,
    *,
    rank: int,
    carry_record: FoldingCarryForwardRecord | None = None,
) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
    dependency_id = _single_predecessor(action)
    assert dependency_id is not None
    predecessor = _load_predecessor_handoff(
        actions_dir / dependency_id / "handoff.json", runspec, dependency_id, "preprocess"
    )
    if not isinstance(predecessor, FoldingPreprocessHandoff):
        raise FoldingExecutorError("fold predecessor must be a preprocess handoff")
    predecessor_digest = canonical_mapping_digest(predecessor.to_mapping())

    backend = runspec.payload.backend
    assets = runspec.cluster.backend_assets
    if assets is None:
        raise FoldingExecutorError("fold cannot execute a historical authority without backend assets")
    if assets.backend != backend:
        raise FoldingExecutorError(f"backend assets declare {assets.backend!r}, not the RunSpec backend {backend!r}")

    if backend == "openfold-trt":
        raise FoldingExecutorError(OPENFOLD_TRT_DEFERRED_MODEL_FN_ERROR)

    packed = action.resources.is_packed
    selection: ShardSelection | None = None
    if packed:
        selection = _load_packed_shard_selection(phase_runspec_path, runspec, action, rank)

    fold_action_digest = ""
    qualification_tuple_id = ""
    carried_target_ids: set[str] = set()
    if packed:
        assert selection is not None
        fold_action_digest = canonical_mapping_digest(action.to_mapping())
        qualification_tuple_id = folding_qualification_tuple_id(
            backend=backend,
            kernel_image=_fold_action_param(action, "kernel_image"),
            cluster_snapshot_digest=canonical_mapping_digest(runspec.cluster.to_mapping()),
        )
        if carry_record is not None:
            binding = _build_carry_adoption_binding(runspec, action, predecessor, selection)
            carried_target_ids = set(
                adopt_carried_outputs(
                    carry_record,
                    successor_action_root=action_root,
                    rank=rank,
                    authority_binding=binding,
                )
            )
        predecessor_by_id = {target.target.target_id: target for target in predecessor.targets}
        ordered_targets: list[FoldingPreprocessTarget] = []
        for target_id in selection.target_ids:
            if target_id in carried_target_ids:
                continue
            target = predecessor_by_id.get(target_id)
            if target is None:
                raise FoldingExecutorError(
                    f"fold shard projection target {target_id!r} is absent from the preprocess predecessor "
                    "(shard/predecessor drift)"
                )
            ordered_targets.append(target)
        outputs_root = action_root / "ranks" / str(rank) / "outputs"
        colabfold_cache_dir = action_root / "ranks" / str(rank) / "colabfold-cache"
        colabfold_structures_dir = action_root / "ranks" / str(rank) / "colabfold-structures"
        journal_path = action_root / "ranks" / str(rank) / "journal.jsonl"
    else:
        ordered_targets = list(predecessor.targets)
        outputs_root = action_root / "outputs"
        colabfold_cache_dir = action_root / "colabfold-cache"
        colabfold_structures_dir = action_root / "colabfold-structures"
        journal_path = None

    session: BioIRFoldSession | None = None
    policy_sessions: BioIRPolicySessions | None = None
    journal: RankJournalWriter | None = None
    try:
        if backend == "openfold-cli":
            if assets.openfold_model_dir is None or assets.chain_manifest_csv is None:
                raise FoldingExecutorError("openfold-cli requires openfold_model_dir and chain_manifest_csv")
            manifest = deps.parse_chain_manifest(Path(assets.chain_manifest_csv))
            fold_backend: OpenFoldCliBackend | ColabFoldBackend | BioIRFoldSession = deps.openfold_backend_factory(
                Path(assets.openfold_model_dir), manifest
            )
        elif backend == "colabfold":
            if assets.colabfold_weights_dir is None or assets.chain_manifest_csv is None:
                raise FoldingExecutorError("colabfold requires colabfold_weights_dir and chain_manifest_csv")
            manifest = deps.parse_chain_manifest(Path(assets.chain_manifest_csv))
            config = ColabFoldConfig(
                weights_dir=Path(assets.colabfold_weights_dir),
                msa_cache_dir=colabfold_cache_dir,
                structures_dir=colabfold_structures_dir,
            )
            fold_backend = deps.colabfold_backend_factory(config, manifest)
        elif backend == "bioir":
            if assets.bioir_checkpoint is None:
                raise FoldingExecutorError("bioir requires bioir_checkpoint")
            policy = runspec.payload.bioir_model_policy
            if policy is None:
                session = deps.bioir_session_factory(Path(assets.bioir_checkpoint), outputs_root)
                fold_backend = session
            else:
                if _fold_action_param(action, "bioir_model_policy_digest") != policy.digest:
                    raise FoldingExecutorError("fold action does not bind its BioIR model policy")
                policy_sessions = BioIRPolicySessions(policy, assets, outputs_root, deps.bioir_session_factory)
        else:
            raise FoldingExecutorError(f"unsupported folding backend: {backend!r}")

        if packed:
            assert journal_path is not None
            journal = RankJournalWriter(journal_path)

        targets: list[FoldingFoldTarget] = []
        for preprocess_target in ordered_targets:
            target_id = preprocess_target.target.target_id
            ptarget = ProteinTarget(
                target_id=target_id,
                description=preprocess_target.target.description,
                chains=preprocess_target.target.chains,
            )
            chain_ids = tuple(f"{target_id}_{chr(ord('A') + index)}" for index in range(len(ptarget.chains)))
            prepared = PreparedInput(
                backend=preprocess_target.layout,
                fasta_dir=Path(preprocess_target.fasta_dir),
                alignment_dir=Path(preprocess_target.alignment_dir),
                template_dir=Path(preprocess_target.template_dir),
                metadata={"chain_ids": list(chain_ids), "layout": preprocess_target.layout},
            )

            result: FoldingResult
            if backend == "openfold-cli":
                classification = classify_target(manifest, target_id)
                if classification == "ambiguous":
                    raise FoldingExecutorError(f"ambiguous chain-manifest classification for {target_id}")
                leaked = classification == "leaked"
                out_dir = outputs_root / target_id
                result = cast("OpenFoldCliBackend", fold_backend).run(
                    ptarget, prepared, out_dir, leaked_homodimer=leaked
                )
            elif backend == "colabfold":
                out_dir = outputs_root / target_id
                result = cast("ColabFoldBackend", fold_backend).run(ptarget, prepared, out_dir)
                if not result.predictions:
                    raise FoldingExecutorError(f"ambiguous chain-manifest classification for {target_id}")
            else:
                if policy_sessions is not None:
                    result = policy_sessions.run(ptarget, prepared)
                else:
                    assert session is not None
                    result = session.run(ptarget, prepared, session.output_dir)

            if len(result.predictions) != 1:
                raise FoldingExecutorError(
                    f"fold backend returned {len(result.predictions)} predictions for {target_id}; expected one"
                )
            prediction = result.predictions[0]
            if runspec.payload.evidence_profile == ARTIFACT_EVIDENCE_PROFILE:
                from .artifact_evidence import read_scores

                policy = runspec.payload.bioir_model_policy
                assert policy is not None
                scores_payload, _ = read_scores(
                    prediction.scores_path,
                    root=action_root,
                    length=sum(map(len, preprocess_target.target.chains)),
                    model_source=policy.model_source_for_chain_count(len(preprocess_target.target.chains)),
                )
            else:
                scores_payload = prediction_scores_payload_from_mapping(_read_json_mapping(prediction.scores_path))
            tool_used = result.metadata.get("tool_used")
            if not isinstance(tool_used, str):
                raise FoldingExecutorError("fold backend metadata is missing a 'tool_used' string")
            pair = PredictionPair(
                model_entity_id=normalize_model_entity_id(target_id),
                tool_used=tool_used,
                structure_path=str(prediction.structure_path),
                scores_path=str(prediction.scores_path),
                scores=scores_payload,
            )
            if not packed:
                targets.append(
                    FoldingFoldTarget(
                        target=preprocess_target.target,
                        pair=pair,
                        model_metadata=dict(result.metadata),
                    )
                )
            if packed:
                assert journal is not None and selection is not None
                journal.append(
                    RankJournalEvent(
                        phase_run_id=runspec.phase_run_id,
                        attempt_id=runspec.attempt_id,
                        rank=rank,
                        fold_action_id=action.action_id,
                        fold_action_digest=fold_action_digest,
                        shard_projection_sha256=selection.sha256,
                        shard_projection_worker_count=selection.worker_count,
                        shard_projection_lpt_version=selection.lpt_version,
                        predecessor_digest=predecessor_digest,
                        target_id=preprocess_target.target.target_id,
                        sequence_sha256=preprocess_target.target.sequence_sha256,
                        description=preprocess_target.target.description,
                        backend=backend,
                        qualification_tuple_id=qualification_tuple_id,
                        outputs=(
                            _file_output(Path(prediction.structure_path)),
                            _file_output(Path(prediction.scores_path)),
                        ),
                    )
                )
                # Packed consumers use authenticated journals. Do not retain
                # completed score matrices until the entire shard finishes.
                del pair, scores_payload
    finally:
        if policy_sessions is not None:
            policy_sessions.close()
        if session is not None:
            session.close()
        if journal is not None:
            journal.close()

    if packed:
        return None, None

    handoff = FoldingFoldHandoff(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        action_id=action.action_id,
        predecessor_digest=predecessor_digest,
        backend=backend,
        targets=tuple(targets),
    )
    evidence: dict[str, object] = {"pairs": [target.pair.to_mapping() for target in targets]}
    return handoff.to_mapping(), evidence


_TOOL_USED_BY_BACKEND: dict[str, str] = {
    "colabfold": VALID_TOOL_USED[0],
    "openfold-trt": VALID_TOOL_USED[1],
    "openfold-cli": VALID_TOOL_USED[2],
    "bioir": VALID_TOOL_USED[3],
}


def _tool_used_for_backend(backend: str) -> str:
    """Return the deterministic ``tool_used`` provenance string for one backend.

    Legacy backends emit one backend-wide ``tool_used`` constant. Explicit
    BioIR model policies instead derive per-target provenance from their sealed
    policy and expanded chain count.
    """
    try:
        return _TOOL_USED_BY_BACKEND[backend]
    except KeyError:
        raise FoldingExecutorError(f"unsupported folding backend: {backend!r}") from None


def _validate_rank_journal_closure(
    runspec: FoldingPhaseRunSpec,
    fold_action: FoldingRuntimeAction,
    preprocess: FoldingPreprocessHandoff,
    projection: FoldShardProjection,
    shard_sha256: str,
    actions_dir: Path,
) -> dict[str, RankJournalEvent]:
    """Validate exact rank/target journal closure against the shard projection.

    Every declared target must have exactly one complete event with the full
    authority tuple and verified output path/size/SHA-256. Missing,
    duplicate, foreign, cross-rank, or mismatched events fail closed (no
    polling).
    """
    preprocess_digest = canonical_mapping_digest(preprocess.to_mapping())
    fold_action_digest = canonical_mapping_digest(fold_action.to_mapping())
    backend = runspec.payload.backend

    preprocess_by_id = {target.target.target_id: target for target in preprocess.targets}
    declared_targets: list[str] = []
    rank_of_target: dict[str, int] = {}
    for rank_row in projection.ranks:
        for target in rank_row.targets:
            rank_of_target[target.target_id] = rank_row.global_rank
            declared_targets.append(target.target_id)

    if set(declared_targets) != set(preprocess_by_id):
        raise FoldingExecutorError("fold shard projection does not cover exactly the preprocess target set")

    events_by_target: dict[str, RankJournalEvent] = {}
    for rank in range(projection.worker_count):
        native_path = actions_dir / fold_action.action_id / "ranks" / str(rank) / "journal.jsonl"
        adopted_path = actions_dir / fold_action.action_id / "ranks" / str(rank) / "adopted.jsonl"
        if not native_path.is_file() and not adopted_path.is_file():
            raise FoldingExecutorError(f"missing rank journal {native_path}")
        rank_events: list[RankJournalEvent] = []
        if native_path.is_file():
            rank_events.extend(read_rank_journal(native_path))
        if adopted_path.is_file():
            rank_events.extend(read_adopted_journal(adopted_path))
        for event in rank_events:
            if event.phase_run_id != runspec.phase_run_id:
                raise FoldingExecutorError(f"rank journal event phase_run_id mismatch in {native_path}")
            if event.attempt_id != runspec.attempt_id:
                raise FoldingExecutorError(f"rank journal event attempt_id mismatch in {native_path}")
            if event.fold_action_id != fold_action.action_id:
                raise FoldingExecutorError(f"rank journal event fold_action_id mismatch in {native_path}")
            if event.fold_action_digest != fold_action_digest:
                raise FoldingExecutorError(f"rank journal event fold_action_digest mismatch in {native_path}")
            if event.shard_projection_sha256 != shard_sha256:
                raise FoldingExecutorError(f"rank journal event shard_projection_sha256 mismatch in {native_path}")
            if event.shard_projection_worker_count != projection.worker_count:
                raise FoldingExecutorError(
                    f"rank journal event shard_projection_worker_count mismatch in {native_path}"
                )
            if event.shard_projection_lpt_version != projection.lpt_version:
                raise FoldingExecutorError(f"rank journal event shard_projection_lpt_version mismatch in {native_path}")
            if event.predecessor_digest != preprocess_digest:
                raise FoldingExecutorError(f"rank journal event predecessor_digest mismatch in {native_path}")
            if (
                runspec.payload.evidence_profile == ARTIFACT_EVIDENCE_PROFILE
                and event.qualification_tuple_id
                != folding_qualification_tuple_id(
                    backend=backend,
                    kernel_image=_fold_action_param(fold_action, "kernel_image"),
                    cluster_snapshot_digest=canonical_mapping_digest(runspec.cluster.to_mapping()),
                )
            ):
                raise FoldingExecutorError("rank journal qualification tuple differs from this RunSpec")
            if event.backend != backend:
                raise FoldingExecutorError(f"rank journal event backend mismatch in {native_path}")
            if event.rank != rank:
                raise FoldingExecutorError(f"rank journal {native_path} contains an event declaring rank {event.rank}")
            if event.target_id not in rank_of_target:
                raise FoldingExecutorError(f"foreign target {event.target_id!r} in rank journal {native_path}")
            if rank_of_target[event.target_id] != event.rank:
                raise FoldingExecutorError(f"cross-rank event for target {event.target_id!r} in {native_path}")
            if event.target_id in events_by_target:
                raise FoldingExecutorError(f"duplicate fold journal event for target {event.target_id!r}")
            preprocess_target = preprocess_by_id[event.target_id]
            if event.sequence_sha256 != preprocess_target.target.sequence_sha256:
                raise FoldingExecutorError(f"rank journal event sequence_sha256 mismatch for {event.target_id}")
            if event.description != preprocess_target.target.description:
                raise FoldingExecutorError(f"rank journal event description mismatch for {event.target_id}")
            if len(event.outputs) != 2:
                raise FoldingExecutorError(f"rank journal event for {event.target_id} must carry exactly two outputs")
            for output in event.outputs:
                output_path = Path(output.path)
                if not output_path.is_file():
                    raise FoldingExecutorError(f"rank journal output missing: {output.path}")
                if runspec.payload.evidence_profile == ARTIFACT_EVIDENCE_PROFILE:
                    from .artifact_evidence import snapshot_file

                    _, size, digest = snapshot_file(
                        output_path,
                        root=actions_dir / fold_action.action_id,
                        maximum_bytes=MAX_SCORE_BYTES,
                        retain=False,
                    )
                    actual = RankJournalOutput(str(output_path), size, digest)
                else:
                    actual = _file_output(output_path)
                if actual.size != output.size or actual.sha256 != output.sha256:
                    raise FoldingExecutorError(f"rank journal output mismatch: {output.path}")
            events_by_target[event.target_id] = event

    if set(events_by_target) != set(declared_targets):
        missing = sorted(set(declared_targets) - set(events_by_target))
        raise FoldingExecutorError(f"missing fold journal events for targets: {', '.join(missing)}")

    return events_by_target


def _reduce_rank_journals(
    phase_runspec_path: Path,
    runspec: FoldingPhaseRunSpec,
    fold_action: FoldingRuntimeAction,
    actions_dir: Path,
) -> FoldingFoldHandoff:
    """Reduce the packed fold action's rank journals into the aggregate fold view.

    After exact closure passes, this reconstructs the ordered
    :class:`FoldingFoldTarget` records in preprocess-handoff (member) order,
    atomically publishes the single aggregate ``FoldActionEvidence`` and the
    ``FoldingFoldHandoff`` projection into the fold action's own paths
    into the fold action's own paths,
    and returns the handoff.
    """
    selection = _load_packed_shard_selection(phase_runspec_path, runspec, fold_action, 0)

    dependency_id = _single_predecessor(fold_action)
    assert dependency_id is not None
    preprocess = _load_predecessor_handoff(
        actions_dir / dependency_id / "handoff.json", runspec, dependency_id, "preprocess"
    )
    if not isinstance(preprocess, FoldingPreprocessHandoff):
        raise FoldingExecutorError("fold predecessor must be a preprocess handoff")

    events_by_target = _validate_rank_journal_closure(
        runspec=runspec,
        fold_action=fold_action,
        preprocess=preprocess,
        projection=selection.projection,
        shard_sha256=selection.sha256,
        actions_dir=actions_dir,
    )

    backend = runspec.payload.backend
    policy = runspec.payload.bioir_model_policy
    if policy is not None and _fold_action_param(fold_action, "bioir_model_policy_digest") != policy.digest:
        raise FoldingExecutorError("fold action does not bind its BioIR model policy")
    targets: list[FoldingFoldTarget] = []
    for preprocess_target in preprocess.targets:
        metadata = (
            bioir_model_metadata(policy, len(preprocess_target.target.chains))
            if policy is not None
            else {"tool_used": _tool_used_for_backend(backend)}
        )
        tool_used = str(metadata["tool_used"])
        event = events_by_target[preprocess_target.target.target_id]
        structure_output = event.outputs[0]
        scores_output = event.outputs[1]
        scores_payload = prediction_scores_payload_from_mapping(_read_json_mapping(Path(scores_output.path)))
        pair = PredictionPair(
            model_entity_id=normalize_model_entity_id(preprocess_target.target.target_id),
            tool_used=tool_used,
            structure_path=structure_output.path,
            scores_path=scores_output.path,
            scores=scores_payload,
        )
        targets.append(
            FoldingFoldTarget(
                target=preprocess_target.target,
                pair=pair,
                model_metadata=metadata,
            )
        )

    fold_handoff = FoldingFoldHandoff(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        action_id=fold_action.action_id,
        predecessor_digest=canonical_mapping_digest(preprocess.to_mapping()),
        backend=backend,
        targets=tuple(targets),
    )
    aggregate_evidence: dict[str, object] = {"pairs": [target.pair.to_mapping() for target in targets]}
    FoldActionEvidence.from_mapping(aggregate_evidence)

    fold_action_root = actions_dir / fold_action.action_id
    _atomic_write_json(fold_action_root / "handoff.json", fold_handoff.to_mapping())
    _atomic_write_json(fold_action_root / "action-evidence.json", aggregate_evidence)
    return fold_handoff


def _dispatch_canonical_pair(
    phase_runspec_path: Path,
    runspec: FoldingPhaseRunSpec,
    action: FoldingRuntimeAction,
    action_root: Path,
    actions_dir: Path,
    deps: FoldingExecutorDependencies,
    *,
    carry_record: FoldingCarryForwardRecord | None = None,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    policy = runspec.payload.bioir_model_policy
    if policy is not None and _fold_action_param(action, "bioir_model_policy_digest") != policy.digest:
        raise FoldingExecutorError("canonical action does not bind its BioIR model policy")
    dependency_id = _single_predecessor(action)
    assert dependency_id is not None
    fold_action = next(candidate for candidate in runspec.payload.actions if candidate.action_id == dependency_id)
    packed = fold_action.resources.is_packed

    if packed:
        if carry_record is not None and runspec.payload.evidence_profile != ARTIFACT_EVIDENCE_PROFILE:
            selection = _load_packed_shard_selection(phase_runspec_path, runspec, fold_action, 0)
            fold_dependency_id = _single_predecessor(fold_action)
            assert fold_dependency_id is not None
            preprocess = _load_predecessor_handoff(
                actions_dir / fold_dependency_id / "handoff.json",
                runspec,
                fold_dependency_id,
                "preprocess",
            )
            if not isinstance(preprocess, FoldingPreprocessHandoff):
                raise FoldingExecutorError("fold predecessor must be a preprocess handoff")
            adoption_binding = _build_carry_adoption_binding(runspec, fold_action, preprocess, selection)
            adopt_all_carried_outputs(
                carry_record,
                successor_action_root=actions_dir / fold_action.action_id,
                authority_binding=adoption_binding,
            )
        if runspec.payload.evidence_profile == ARTIFACT_EVIDENCE_PROFILE:
            from .artifact_evidence import build_canonical_evidence, prepare_carry_journals

            selection = _load_packed_shard_selection(phase_runspec_path, runspec, fold_action, 0)
            fold_dependency = _single_predecessor(fold_action)
            assert fold_dependency is not None
            preprocess = _load_predecessor_handoff(
                actions_dir / fold_dependency / "handoff.json", runspec, fold_dependency, "preprocess"
            )
            if not isinstance(preprocess, FoldingPreprocessHandoff):
                raise FoldingExecutorError("artifact fold predecessor must be a preprocess handoff")
            prepare_carry_journals(
                runspec=runspec,
                fold_root=actions_dir / fold_action.action_id,
                projection=selection.projection,
                record=carry_record,
                binding=_build_carry_adoption_binding(runspec, fold_action, preprocess, selection),
            )
            events = _validate_rank_journal_closure(
                runspec, fold_action, preprocess, selection.projection, selection.sha256, actions_dir
            )
            return build_canonical_evidence(
                runspec=runspec,
                action=action,
                fold_action=fold_action,
                preprocess=preprocess,
                events=events,
                actions_dir=actions_dir,
                provenance=_with_install_provenance,
            )
        fold_handoff = _reduce_rank_journals(phase_runspec_path, runspec, fold_action, actions_dir)
        predecessor_digest = canonical_mapping_digest(fold_handoff.to_mapping())
        fold_targets = fold_handoff.targets
    else:
        predecessor = _load_predecessor_handoff(
            actions_dir / dependency_id / "handoff.json", runspec, dependency_id, "fold"
        )
        if not isinstance(predecessor, FoldingFoldHandoff):
            raise FoldingExecutorError("canonical-pair predecessor must be a fold handoff")
        predecessor_digest = canonical_mapping_digest(predecessor.to_mapping())
        fold_targets = predecessor.targets

    canonical_entries: list[dict[str, object]] = []
    index_entries: list[tuple[str, str, str, str, str, str]] = []
    handoff_entries: list[FoldingCanonicalPairEntry] = []
    for fold_target in fold_targets:
        target_id = fold_target.target.target_id
        pair = fold_target.pair
        canonical_entries.append(
            {
                "target_id": target_id,
                "sequence_sha256": fold_target.target.sequence_sha256,
                "pair": pair.to_mapping(),
            }
        )
        index_entries.append(
            (
                target_id,
                fold_target.target.sequence_sha256,
                pair.model_entity_id,
                pair.tool_used,
                pair.structure_path,
                pair.scores_path,
            )
        )
        handoff_entries.append(
            FoldingCanonicalPairEntry(
                target_id=target_id,
                sequence_sha256=fold_target.target.sequence_sha256,
                model_entity_id=pair.model_entity_id,
                tool_used=pair.tool_used,
                structure_path=pair.structure_path,
                scores_path=pair.scores_path,
            )
        )

    from bspp.orchestration.runtime.folding.benchmark.index import (
        build_canonical_pair_index,
        write_canonical_pair_index,
    )

    index_path = action_root / "canonical-pair-index.json"
    index = build_canonical_pair_index(run_id=runspec.phase_run_id, entries=index_entries)
    write_canonical_pair_index(index, index_path)
    index_digest = hashlib.sha256(index.to_json().encode()).hexdigest()

    handoff = FoldingCanonicalPairHandoff(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        action_id=action.action_id,
        predecessor_digest=predecessor_digest,
        index_path=str(index_path),
        index_digest=index_digest,
        entries=tuple(handoff_entries),
    )

    canonical_evidence: dict[str, object] = {"entries": canonical_entries}
    combined: dict[str, object] = {}
    for candidate in runspec.payload.actions:
        if candidate.action_kind == "canonical-pair":
            CanonicalPairActionEvidence.from_mapping(canonical_evidence)
            combined[candidate.action_id] = canonical_evidence
            continue
        evidence_path = actions_dir / candidate.action_id / "action-evidence.json"
        payload = _read_json_mapping(evidence_path)
        if candidate.action_kind == "msa-flatten":
            MsaFlattenActionEvidence.from_mapping(payload)
        elif candidate.action_kind == "split":
            SplitActionEvidence.from_mapping(payload)
        elif candidate.action_kind == "preprocess":
            PreprocessActionEvidence.from_mapping(payload)
        elif candidate.action_kind == "fold":
            FoldActionEvidence.from_mapping(payload)
        else:
            raise FoldingExecutorError(f"unsupported folding action kind: {candidate.action_kind!r}")
        combined[candidate.action_id] = payload

    return handoff.to_mapping(), combined


def _dispatch(
    phase_runspec_path: Path,
    runspec: FoldingPhaseRunSpec,
    action: FoldingRuntimeAction,
    action_root: Path,
    actions_dir: Path,
    deps: FoldingExecutorDependencies,
    *,
    rank: int,
    carry_record: FoldingCarryForwardRecord | None = None,
) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
    if action.action_kind != "fold" and rank != 0:
        raise FoldingExecutorError(f"non-fold action {action.action_id} cannot execute with rank {rank}")
    kind = action.action_kind
    if kind == "msa-flatten":
        return _dispatch_msa_flatten(runspec, action, action_root, deps)
    if kind == "split":
        return _dispatch_split(runspec, action, action_root, actions_dir, deps)
    if kind == "preprocess":
        return _dispatch_preprocess(runspec, action, action_root, actions_dir, deps)
    if kind == "fold":
        return _dispatch_fold(
            phase_runspec_path, runspec, action, action_root, actions_dir, deps, rank=rank, carry_record=carry_record
        )
    if kind == "canonical-pair":
        return _dispatch_canonical_pair(
            phase_runspec_path, runspec, action, action_root, actions_dir, deps, carry_record=carry_record
        )
    raise FoldingExecutorError(f"unsupported folding action kind: {kind!r}")


def run_execute_action(
    *,
    phase_runspec_path: Path,
    action_id: str,
    action_evidence_path: Path,
    handoff_path: Path,
    deps: FoldingExecutorDependencies | None = None,
    rank: int = 0,
    carry_record_path: Path | None = None,
) -> Mapping[str, object]:
    """Execute one exact folding Runtime Action and publish its handoff/evidence.

    ``rank`` selects one global-rank row of the canonical shard projection for a
    packed fold action (default 0 keeps the legacy scalar path byte-for-byte).
    ``carry_record_path`` names the staged sealed carry record the renderer
    supplies whenever the RunSpec carries a reference; it is strict-loaded and
    adopted before any successor fold dispatch.
    """
    if deps is None:
        deps = PRODUCTION_EXECUTOR_DEPS
    try:
        return _run_execute_action(
            phase_runspec_path=phase_runspec_path,
            action_id=action_id,
            action_evidence_path=action_evidence_path,
            handoff_path=handoff_path,
            deps=deps,
            rank=rank,
            carry_record_path=carry_record_path,
        )
    except FoldingExecutorError:
        raise
    except (OSError, TypeError, ValueError, FoldingBackendError) as exc:
        raise FoldingExecutorError(str(exc)) from exc


_VALID_INSTALL_MODES = frozenset({"override", "baked"})


def _with_install_provenance(
    handoff_mapping: dict[str, object],
) -> dict[str, object]:
    """Inject install-mode provenance from environment into the handoff mapping.

    Only truthy values are injected: an empty ``BSPP_ORCHESTRATION_PROVENANCE_COMMIT``
    is never written. ``BSPP_ORCHESTRATION_SOURCE`` is validated against the
    allowed set when non-empty.
    """
    install_mode = os.environ.get("BSPP_ORCHESTRATION_SOURCE")
    source_commit = os.environ.get("BSPP_ORCHESTRATION_PROVENANCE_COMMIT")
    if install_mode:
        if install_mode not in _VALID_INSTALL_MODES:
            raise FoldingExecutorError(
                f"invalid BSPP_ORCHESTRATION_SOURCE={install_mode!r}; expected 'override' or 'baked'"
            )
        handoff_mapping["install_mode"] = install_mode
    if source_commit:
        handoff_mapping["orchestration_source_commit"] = source_commit
    return handoff_mapping


def _run_execute_action(
    *,
    phase_runspec_path: Path,
    action_id: str,
    action_evidence_path: Path,
    handoff_path: Path,
    deps: FoldingExecutorDependencies,
    rank: int = 0,
    carry_record_path: Path | None = None,
) -> Mapping[str, object]:
    if handoff_path.exists() or action_evidence_path.exists():
        raise FoldingExecutorError("action already completed; rerun into a nonempty successful action root is rejected")

    runspec = deps.load_runspec(phase_runspec_path)

    matches = [action for action in runspec.payload.actions if action.action_id == action_id]
    if len(matches) != 1:
        raise FoldingExecutorError(f"expected exactly one action with id {action_id!r}; found {len(matches)}")
    action = matches[0]

    carry_record: FoldingCarryForwardRecord | None = None
    # The public renderer stages carry bytes only for their consumers. Preserve
    # the historical guard on v1; v2 prerequisites need no unmounted carry file.
    consumes_carry = runspec.payload.evidence_profile != ARTIFACT_EVIDENCE_PROFILE or action.action_kind in {
        "fold",
        "canonical-pair",
    }
    if runspec.carry_forward is not None and consumes_carry:
        if carry_record_path is None:
            raise FoldingExecutorError("carried folding RunSpec requires --carry-record")
        carry_record = _load_carry_record(carry_record_path)
        if runspec.payload.evidence_profile == ARTIFACT_EVIDENCE_PROFILE:
            from .artifact_evidence import validate_carry_reference

            validate_carry_reference(runspec, carry_record)

    action_root, actions_dir, _ = _validate_handoff_path_identity(handoff_path, action_id)
    _validate_evidence_path_confinement(action_evidence_path, action_root)
    _validate_predecessor_action_ids(action, runspec)
    packed = action.action_kind == "fold" and action.resources.is_packed
    _reject_partial_rerun_content(action_root, rank=rank, packed=packed)

    handoff_mapping, evidence_mapping = _dispatch(
        phase_runspec_path, runspec, action, action_root, actions_dir, deps, rank=rank, carry_record=carry_record
    )
    if handoff_mapping is None and evidence_mapping is None:
        journal_path = action_root / "ranks" / str(rank) / "journal.jsonl"
        events = read_rank_journal(journal_path)
        return {
            "rank": rank,
            "rank_journal": str(journal_path),
            "folded_targets": [event.target_id for event in events],
        }
    assert handoff_mapping is not None and evidence_mapping is not None
    handoff_mapping = _with_install_provenance(dict(handoff_mapping))
    if runspec.payload.evidence_profile == ARTIFACT_EVIDENCE_PROFILE and action.action_kind == "canonical-pair":
        from .artifact_evidence import publish_finalization_index, write_metadata

        write_metadata(handoff_path, handoff_mapping)
        write_metadata(action_evidence_path, evidence_mapping)
        publish_finalization_index(runspec, actions_dir, handoff_mapping)
    else:
        _atomic_write_json(handoff_path, handoff_mapping)
        _atomic_write_json(action_evidence_path, evidence_mapping)
    return handoff_mapping


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the rendered folding action argv and delegate to the executor."""
    parser = argparse.ArgumentParser(prog="python -m bspp.orchestration.runtime.folding.executor")
    parser.add_argument("--phase-runspec", required=True, type=Path)
    parser.add_argument("--action-id", required=True)
    parser.add_argument("--action-evidence", required=True, type=Path)
    parser.add_argument("--handoff", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--carry-record", type=Path, default=None)
    args = parser.parse_args(argv)
    handoff = run_execute_action(
        phase_runspec_path=args.phase_runspec,
        action_id=args.action_id,
        action_evidence_path=args.action_evidence,
        handoff_path=args.handoff,
        deps=PRODUCTION_EXECUTOR_DEPS,
        rank=args.rank,
        carry_record_path=args.carry_record,
    )
    print(json.dumps(handoff, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1:])


__all__ = [
    "PRODUCTION_EXECUTOR_DEPS",
    "FoldingExecutorDependencies",
    "FoldingExecutorError",
    "main",
    "run_execute_action",
]
