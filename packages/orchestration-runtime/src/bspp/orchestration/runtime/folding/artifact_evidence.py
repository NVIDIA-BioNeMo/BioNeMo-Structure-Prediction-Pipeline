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

"""Job-local bounded evidence generation from original authenticated score files."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath

from bspp.orchestration.contract.folding_artifact_evidence import (
    MAX_METADATA_BYTES,
    MAX_SCORE_BYTES,
    ArtifactFoldEvidence,
    ArtifactFoldTarget,
    FoldingFinalizationIndex,
    FoldingMetadataMember,
    PredictionArtifactIdentity,
    finalization_member_paths,
    model_metadata,
)
from bspp.orchestration.contract.folding_carry_forward import FoldingCarryForwardRecord
from bspp.orchestration.contract.folding_evidence import (
    MsaFlattenActionEvidence,
    PreprocessActionEvidence,
    SplitActionEvidence,
)
from bspp.orchestration.contract.folding_execution import (
    FoldingCanonicalPairEntry,
    FoldingCanonicalPairHandoff,
    FoldingPreprocessHandoff,
)
from bspp.orchestration.contract.folding_shard import FoldShardProjection
from bspp.orchestration.contract.phase import FoldingPhaseRunSpec, FoldingRuntimeAction, canonical_mapping_digest
from bspp.orchestration.contract.prediction_pair import PredictionScoresPayload, prediction_scores_payload_from_mapping

from .carry_adoption import CarryAdoptionAuthorityBinding, adopt_all_carried_outputs
from .rank_journal import RankJournalEvent, RankJournalWriter


def _signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _open_regular(path: Path, root: Path) -> int:
    """Open through directory descriptors, refusing every symlink ancestor."""
    if not path.is_absolute() or not root.is_absolute() or not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError("prediction artifact escapes its owned root")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = following
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("prediction artifact is not a regular file")
    return descriptor


def snapshot_file(path: Path, *, root: Path, maximum_bytes: int, retain: bool = True) -> tuple[bytes, int, str]:
    """Hash and optionally retain exactly one bounded, stable regular snapshot."""
    descriptor = _open_regular(path, root)
    chunks: list[bytes] = []
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not 0 < before.st_size <= maximum_bytes:
            raise ValueError("artifact snapshot exceeds its fixed size bound")
        digest = hashlib.sha256()
        size = 0
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError("growing artifact snapshot exceeds its fixed size bound")
            digest.update(chunk)
            if retain:
                chunks.append(chunk)
        after = os.fstat(handle.fileno())
        reopened = _open_regular(path, root)
        try:
            current = os.fstat(reopened)
        finally:
            os.close(reopened)
        if (
            size != before.st_size
            or _signature(before) != _signature(after)
            or _signature(after) != _signature(current)
        ):
            raise ValueError("artifact changed while its snapshot was read")
    return b"".join(chunks), size, digest.hexdigest()


def json_mapping(data: bytes) -> Mapping[str, object]:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate field in artifact JSON")
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise ValueError(f"nonfinite constant in artifact JSON: {value}")

    value = json.loads(data, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    if not isinstance(value, Mapping):
        raise ValueError("artifact JSON must contain an object")
    return value


def read_scores(
    path: Path, *, root: Path, length: int, model_source: str
) -> tuple[PredictionScoresPayload, PredictionArtifactIdentity]:
    data, size, digest = snapshot_file(path, root=root, maximum_bytes=MAX_SCORE_BYTES)
    scores = prediction_scores_payload_from_mapping(json_mapping(data))
    if scores.extras.get("bioir_model_source") != model_source:
        raise ValueError("original BioIR score model source differs from the sealed policy")
    if len(scores.plddt) != length or len(scores.pae) != length or any(len(row) != length for row in scores.pae):
        raise ValueError("full PAE/plddt dimensions differ from the expanded target length")
    return scores, PredictionArtifactIdentity("scores", str(path), size, digest)


def iter_metadata_bytes(payload: Mapping[str, object]) -> Iterator[bytes]:
    """The bounded exact serializer shared by publication and size qualification."""
    total = 0
    for fragment in json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False).iterencode(payload):
        encoded = fragment.encode("utf-8")
        total += len(encoded)
        if total + 1 > MAX_METADATA_BYTES:
            raise ValueError("folding metadata exceeds its fixed size bound")
        yield encoded
    yield b"\n"


def write_metadata(path: Path, payload: Mapping[str, object]) -> None:
    """Create one bounded metadata file without replacing existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            for encoded in iter_metadata_bytes(payload):
                handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def build_canonical_evidence(
    *,
    runspec: FoldingPhaseRunSpec,
    action: FoldingRuntimeAction,
    fold_action: FoldingRuntimeAction,
    preprocess: FoldingPreprocessHandoff,
    events: Mapping[str, RankJournalEvent],
    actions_dir: Path,
    provenance: Callable[[dict[str, object]], dict[str, object]],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    policy = runspec.payload.bioir_model_policy
    projection = runspec.payload.fold_shard_projection
    if policy is None or projection is None:
        raise ValueError("artifact-backed folding requires policy and projection")
    fold_root = actions_dir / fold_action.action_id
    entries: list[ArtifactFoldTarget] = []
    for prepared in preprocess.targets:
        target = prepared.target
        event = events[target.target_id]
        structure, score_output = event.outputs
        _, structure_size, structure_sha = snapshot_file(
            Path(structure.path), root=fold_root, maximum_bytes=MAX_SCORE_BYTES, retain=False
        )
        scores, score_identity = read_scores(
            Path(score_output.path),
            root=fold_root,
            length=sum(map(len, target.chains)),
            model_source=policy.model_source_for_chain_count(len(target.chains)),
        )
        if (structure_size, structure_sha) != (structure.size, structure.sha256) or (
            score_identity.size_bytes,
            score_identity.sha256,
        ) != (score_output.size, score_output.sha256):
            raise ValueError("artifact snapshot differs from its durable completion journal")
        meta = model_metadata(policy, len(target.chains))
        entry = ArtifactFoldTarget.from_mapping(
            {
                "target": target.to_mapping(),
                "pair_reference": {
                    "model_entity_id": target.target_id,
                    "tool_used": meta["tool_used"],
                    "artifacts": [
                        PredictionArtifactIdentity(
                            "structure", structure.path, structure_size, structure_sha
                        ).to_mapping(),
                        score_identity.to_mapping(),
                    ],
                    "model_metadata": meta,
                    "content_validation": {
                        "profile": "prediction-scores-full-pae-v1",
                        "outcome": "passed",
                        "residue_count": len(scores.plddt),
                        "plddt_count": len(scores.plddt),
                        "pae_rows": len(scores.pae),
                        "pae_columns": len(scores.pae[0]),
                    },
                },
            }
        )
        entries.append(entry)
        del scores
    fold = ArtifactFoldEvidence(
        runspec.phase_run_id,
        runspec.attempt_id,
        runspec.digest,
        fold_action.action_id,
        canonical_mapping_digest(fold_action.to_mapping()),
        canonical_mapping_digest(preprocess.to_mapping()),
        projection.sha256,
        tuple(entries),
    )
    fold = ArtifactFoldEvidence.from_mapping(provenance(fold.to_mapping()))
    fold.validate_binding(runspec, actions_root=PurePosixPath(actions_dir))
    write_metadata(fold_root / "handoff.json", fold.to_mapping())
    write_metadata(fold_root / "action-evidence.json", fold.to_mapping())
    canonical = replace(
        fold,
        action_id=action.action_id,
        action_digest=canonical_mapping_digest(action.to_mapping()),
        predecessor_digest=canonical_mapping_digest(fold.to_mapping()),
    )
    canonical.validate_binding(runspec, actions_root=PurePosixPath(actions_dir))
    from .benchmark.index import build_canonical_pair_index

    index = build_canonical_pair_index(
        run_id=runspec.phase_run_id,
        entries=[
            (
                entry.target.target_id,
                entry.target.sequence_sha256,
                entry.model_entity_id,
                entry.tool_used,
                entry.structure.path,
                entry.scores.path,
            )
            for entry in entries
        ],
    )
    index_path = actions_dir / action.action_id / "canonical-pair-index.json"
    # Preserve the canonical index serializer byte-for-byte.
    write_metadata(index_path, json_mapping(index.to_json().encode()))
    handoff = FoldingCanonicalPairHandoff(
        phase_run_id=runspec.phase_run_id,
        attempt_id=runspec.attempt_id,
        action_id=action.action_id,
        predecessor_digest=canonical.predecessor_digest,
        index_path=str(index_path),
        index_digest=hashlib.sha256(index.to_json().encode()).hexdigest(),
        entries=tuple(
            FoldingCanonicalPairEntry(
                entry.target.target_id,
                entry.target.sequence_sha256,
                entry.model_entity_id,
                entry.tool_used,
                entry.structure.path,
                entry.scores.path,
            )
            for entry in entries
        ),
    )
    combined: dict[str, object] = {}
    for candidate in runspec.payload.actions:
        if candidate.action_kind == "canonical-pair":
            combined[candidate.action_id] = canonical.to_mapping()
        elif candidate.action_kind == "fold":
            combined[candidate.action_id] = fold.to_mapping()
        else:
            data, _, _ = snapshot_file(
                actions_dir / candidate.action_id / "action-evidence.json",
                root=actions_dir,
                maximum_bytes=MAX_METADATA_BYTES,
            )
            payload = json_mapping(data)
            if candidate.action_kind == "msa-flatten":
                MsaFlattenActionEvidence.from_mapping(payload)
            elif candidate.action_kind == "split":
                SplitActionEvidence.from_mapping(payload)
            elif candidate.action_kind == "preprocess":
                PreprocessActionEvidence.from_mapping(payload)
            else:
                raise ValueError("unexpected folding action kind")
            combined[candidate.action_id] = payload
    return handoff.to_mapping(), combined


def publish_finalization_index(runspec: FoldingPhaseRunSpec, actions_dir: Path, handoff: Mapping[str, object]) -> None:
    fold = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    canonical = next(action for action in runspec.payload.actions if action.action_kind == "canonical-pair")
    members = []
    for relative in finalization_member_paths(fold.action_id, canonical.action_id):
        _, size, digest = snapshot_file(
            actions_dir / relative, root=actions_dir, maximum_bytes=MAX_METADATA_BYTES, retain=False
        )
        members.append(FoldingMetadataMember(relative, size, digest))
    source = handoff.get("orchestration_source_commit")
    if not isinstance(source, str):
        raise ValueError("artifact finalization requires installed source provenance")
    index = FoldingFinalizationIndex(
        runspec.phase_run_id,
        runspec.attempt_id,
        runspec.digest,
        fold.action_id,
        canonical_mapping_digest(fold.to_mapping()),
        canonical.action_id,
        canonical_mapping_digest(canonical.to_mapping()),
        source,
        tuple(members),
    )
    index.validate_binding(runspec)
    write_metadata(actions_dir / canonical.action_id / "finalization-index.json", index.to_mapping())


def validate_carry_reference(runspec: FoldingPhaseRunSpec, record: FoldingCarryForwardRecord) -> None:
    reference = runspec.carry_forward
    if (
        reference is None
        or (reference.folding_carry_forward_id, reference.digest) != (record.folding_carry_forward_id, record.digest)
        or (record.phase_run_id, record.phase_plan_digest, record.target_attempt_id, record.backend)
        != (runspec.phase_run_id, runspec.phase_plan_digest, runspec.attempt_id, runspec.payload.backend)
    ):
        raise ValueError("artifact carry record differs from its sealed RunSpec reference")


def prepare_carry_journals(
    *,
    runspec: FoldingPhaseRunSpec,
    fold_root: Path,
    projection: FoldShardProjection,
    record: FoldingCarryForwardRecord | None,
    binding: CarryAdoptionAuthorityBinding,
) -> None:
    """Adopt a skipped full-carry action once, or verify workers' existing adoption."""
    carry_by_id = {} if record is None else {item.target_id: item for item in record.content}
    projected = {item.target_id: rank.global_rank for rank in projection.ranks for item in rank.targets}
    if record is not None:
        validate_carry_reference(runspec, record)
        if any(
            item.target_id not in projected or item.source_rank != projected[item.target_id] for item in record.content
        ):
            raise ValueError("artifact carry does not match the frozen projection")
    journals = [
        fold_root / "ranks" / str(rank) / name
        for rank in range(projection.worker_count)
        for name in ("journal.jsonl", "adopted.jsonl")
    ]
    if record is not None and not any(path.exists() or path.is_symlink() for path in journals):
        if set(carry_by_id) != set(projected):
            raise ValueError("canonical-only adoption requires complete projected carry")
        adopt_all_carried_outputs(record, successor_action_root=fold_root, authority_binding=binding)
        # Empty ranks performed no science. The ordinary empty journal records
        # only their closed zero-target assignment, without a completion event.
        for rank_row in projection.ranks:
            if not rank_row.targets:
                writer = RankJournalWriter(fold_root / "ranks" / str(rank_row.global_rank) / "journal.jsonl")
                writer.close()
    observed: set[str] = set()
    for rank in range(projection.worker_count):
        path = fold_root / "ranks" / str(rank) / "adopted.jsonl"
        if not path.exists() and not path.is_symlink():
            continue
        data, _, _ = snapshot_file(path, root=fold_root, maximum_bytes=MAX_METADATA_BYTES)
        if not data.endswith(b"\n"):
            raise ValueError("artifact adopted journal has an incomplete final event")
        for line in data.splitlines():
            payload = json_mapping(line)
            target_id = payload.get("target_id")
            if not isinstance(target_id, str) or target_id not in carry_by_id or target_id in observed:
                raise ValueError("artifact adopted journal has a foreign or duplicate target")
            assert record is not None
            item = carry_by_id[target_id]
            expected_outputs = [
                {
                    "path": str(
                        fold_root / "ranks" / str(rank) / "outputs" / target_id / Path(output.output_path).name
                    ),
                    "size": output.size_bytes,
                    "sha256": output.sha256,
                }
                for output in item.outputs
            ]
            if (
                payload.get("event_kind"),
                payload.get("source_attempt_id"),
                payload.get("carry_record_digest"),
                payload.get("sequence_sha256"),
                rank,
            ) != (
                "adopted",
                record.source_attempt_id,
                record.digest,
                item.sequence_sha256,
                item.source_rank,
            ) or payload.get("outputs") != expected_outputs:
                raise ValueError("artifact adopted journal differs from its sealed carry provenance")
            observed.add(target_id)
    if observed != set(carry_by_id):
        raise ValueError("artifact adopted journals do not cover the exact sealed carry set")
