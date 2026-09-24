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

"""Generic tiny actual-file fixture for artifact-backed evidence (no inference)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from tests.test_folding.test_executor import (
    _ACTION_IDS,
    _FIXTURE_DIR,
    _TARGET_ID,
    RecordingBioIRSession,
    _make_local_location,
    _make_packed_runspec,
    _RecordingDeps,
)

from bspp.orchestration.contract.folding_bioir import BioIRModelPolicy
from bspp.orchestration.contract.folding_execution import FoldingBackendAssetsSnapshot
from bspp.orchestration.contract.folding_shard import (
    fold_shard_projection_document_bytes,
    fold_shard_projection_from_mapping,
)
from bspp.orchestration.contract.preprocessing_handoff import msa_artifact_set_id
from bspp.orchestration.runtime.folding import executor


def fixture(tmp_path: Path, *, workers: int = 2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    runspec, _ = _make_packed_runspec(
        tmp_path,
        backend="bioir",
        chain_manifest_csv=_FIXTURE_DIR / "chain-manifest.csv",
        worker_count=workers,
        rank_target_ids={rank: (_TARGET_ID,) if rank == 0 else () for rank in range(workers)},
        nodes=2 if workers < 16 else workers // 8,
        tasks_per_node=1 if workers < 16 else 8,
    )
    manifest = runspec.payload.msa_set_manifest
    assert manifest is not None
    manifest = replace(
        manifest,
        member_lengths=(4,),
        artifact_set_id=msa_artifact_set_id(
            manifest.chunks, manifest.member_count, manifest.logical_bytes, member_lengths=(4,)
        ),
    )
    location = _make_local_location(tmp_path, manifest, (_FIXTURE_DIR / "merged_compound.a3m").read_bytes())
    mono, multi = tmp_path / "monomer.pt", tmp_path / "multimer.pt"
    mono.write_bytes(b"test-only monomer checkpoint")
    multi.write_bytes(b"test-only multimer checkpoint")
    policy = BioIRModelPolicy(
        hashlib.sha256(mono.read_bytes()).hexdigest(),
        mono.stat().st_size,
        hashlib.sha256(multi.read_bytes()).hexdigest(),
        multi.stat().st_size,
    )
    actions = tuple(
        replace(
            action,
            payload=replace(
                action.payload,
                params=(
                    *action.payload.params,
                    ("bioir_model_policy_digest", policy.digest),
                    ("evidence_profile", "artifact-backed-v2"),
                ),
            ),
        )
        if action.action_kind in {"fold", "canonical-pair"}
        else action
        for action in runspec.payload.actions
    )
    binding = runspec.payload.fold_shard_projection
    assert binding is not None
    projection_path = tmp_path / "fold-shard-projection.json"
    projection = json.loads(projection_path.read_bytes())
    for rank in projection["ranks"]:
        for target in rank["targets"]:
            target["member_length"] = 4
    document = fold_shard_projection_document_bytes(fold_shard_projection_from_mapping(projection))
    projection_path.write_bytes(document)
    runspec = replace(
        runspec,
        input_location=location,
        payload=replace(
            runspec.payload,
            msa_set_manifest=manifest,
            msa_set=replace(runspec.payload.msa_set, artifact_set_id=manifest.artifact_set_id),
            bioir_model_policy=policy,
            evidence_profile="artifact-backed-v2",
            actions=actions,
            fold_shard_projection=replace(
                binding, sha256=hashlib.sha256(document).hexdigest(), size_bytes=len(document)
            ),
        ),
        cluster=replace(
            runspec.cluster,
            project_root=str(tmp_path / "project"),
            backend_assets=FoldingBackendAssetsSnapshot(
                backend="bioir", bioir_checkpoint=str(multi), bioir_monomer_checkpoint=str(mono)
            ),
        ),
    )

    class Session(RecordingBioIRSession):
        def __init__(self, checkpoint, output_dir, *, model_source):
            super().__init__(checkpoint, output_dir)
            self.model_source = model_source

        def run(self, target, prepared, output_dir):
            result = super().run(target, prepared, output_dir)
            scores_path = result.predictions[0].scores_path
            scores = json.loads(scores_path.read_bytes())
            scores["bioir_model_source"] = self.model_source
            scores["preserved_extra"] = {"unicode": "\u03b1", "signed_zero": -0.0}
            scores_path.write_text(json.dumps(scores) + "\n")
            return replace(
                result,
                metadata={
                    "tool_used": "OpenFold2 (BioNeMo IR) / AlphaFold-Multimer",
                    "model_source": self.model_source,
                },
            )

    deps = replace(_RecordingDeps(runspec, "bioir").deps(), bioir_session_factory=Session)
    return runspec, deps


def action_root(runspec, action_id):
    return (
        Path(runspec.cluster.project_root)
        / "bspp-phase-runs"
        / runspec.phase_run_id
        / runspec.attempt_id
        / "actions"
        / action_id
    )


def run_action(tmp_path, runspec, deps, kind, *, rank=0, carry_record_path=None):
    action_id = _ACTION_IDS[kind]
    root = action_root(runspec, action_id)
    root.mkdir(parents=True, exist_ok=True)
    executor.run_execute_action(
        phase_runspec_path=tmp_path / "runspec.json",
        action_id=action_id,
        action_evidence_path=root / "action-evidence.json",
        handoff_path=root / "handoff.json",
        deps=deps,
        rank=rank,
        carry_record_path=carry_record_path,
    )
    return root


def run_to_fold(tmp_path, runspec, deps):
    for kind in ("msa-flatten", "split", "preprocess"):
        run_action(tmp_path, runspec, deps, kind)
    action = next(action for action in runspec.payload.actions if action.action_kind == "fold")
    for rank in range(action.resources.workers):
        run_action(tmp_path, runspec, deps, "fold", rank=rank)
    return action_root(runspec, action.action_id)


def successor_with_carry(tmp_path, runspec, deps, fold_root):
    """Make a genuine test carry record from the just-written rank journal."""
    from bspp.orchestration.contract.folding_carry_forward import (
        FoldingCarryForwardReference,
        folding_carry_forward_id,
        folding_carry_forward_record_from_mapping,
    )
    from bspp.orchestration.contract.phase import canonical_mapping_digest
    from bspp.orchestration.runtime.folding.rank_journal import read_rank_journal

    event = read_rank_journal(fold_root / "ranks/0/journal.jsonl")[0]
    content = [
        {
            "schema_version": 1,
            "target_id": event.target_id,
            "sequence_sha256": event.sequence_sha256,
            "source_action_id": event.fold_action_id,
            "source_rank": event.rank,
            "outputs": [
                {"schema_version": 1, "output_path": output.path, "size_bytes": output.size, "sha256": output.sha256}
                for output in event.outputs
            ],
        }
    ]
    identity = {
        "schema_version": 1,
        "phase_run_id": runspec.phase_run_id,
        "phase_plan_digest": runspec.phase_plan_digest,
        "source_attempt_id": runspec.attempt_id,
        "source_attempt_ordinal": 1,
        "source_runspec_digest": runspec.digest,
        "target_attempt_id": "attempt-0002",
        "target_attempt_ordinal": 2,
        "backend": "bioir",
        "content": content,
        "ancestor_closure": [],
        "content_digest": canonical_mapping_digest({"schema_version": 1, "content": content}),
        "declared_at": "2026-09-19T12:00:00Z",
    }
    record = folding_carry_forward_record_from_mapping(
        {"folding_carry_forward_id": folding_carry_forward_id(identity), **identity}
    )
    path = tmp_path / "folding-carry-forward.json"
    path.write_text(json.dumps(record.to_mapping()))
    successor = replace(
        runspec,
        attempt_id="attempt-0002",
        carry_forward=FoldingCarryForwardReference(
            record.folding_carry_forward_id, record.digest, "attempts/attempt-0002/folding-carry-forward.json"
        ),
    )
    return successor, replace(deps, load_runspec=lambda _: successor), path


def add_second_target(tmp_path, runspec, deps):
    """Extend only generic test inputs with a second real A3M in the same rank."""
    from tests.test_folding.test_executor import _SECOND_LOGICAL_PATH, _SECOND_TARGET_ID, _write_tar

    from bspp.orchestration.contract.preprocessing_handoff import verified_local_bundled_artifact_location_id

    second_member_name = Path(_SECOND_LOGICAL_PATH).name
    old = runspec.input_location
    first = old.members[0]
    second = replace(
        first,
        logical_path=f"a3ms/{second_member_name}",
        member_name=second_member_name,
        raw_member_name=f"./{second_member_name}",
    )
    data = (_FIXTURE_DIR / "merged_compound.a3m").read_bytes()
    raw, digest, size, raw_names = _write_tar([(first.raw_member_name, data), (second.raw_member_name, data)])
    Path(old.tar_path).write_bytes(raw)
    manifest = runspec.payload.msa_set_manifest
    chunks = (replace(manifest.chunks[0], member_count=2, logical_bytes=200),)
    manifest = replace(
        manifest,
        chunks=chunks,
        member_count=2,
        logical_bytes=200,
        member_lengths=(4, 4),
        artifact_set_id=msa_artifact_set_id(chunks, 2, 200, member_lengths=(4, 4)),
    )
    identity = dict(
        artifact_set_id=manifest.artifact_set_id,
        tar_path=old.tar_path,
        bundle_path=old.bundle_path,
        bundle_uri=old.bundle_uri,
        tar_size_bytes=size,
        tar_sha256=digest,
        lz4_size_bytes=old.lz4_size_bytes,
        lz4_sha256=old.lz4_sha256,
        raw_tar_members=raw_names,
        members=(first, second),
    )
    location = replace(old, artifact_location_id=verified_local_bundled_artifact_location_id(**identity), **identity)
    path = tmp_path / "fold-shard-projection.json"
    mapping = json.loads(path.read_bytes())
    mapping["ranks"][0]["targets"].append({"schema_version": 1, "target_id": _SECOND_TARGET_ID, "member_length": 4})
    document = fold_shard_projection_document_bytes(fold_shard_projection_from_mapping(mapping))
    path.write_bytes(document)
    runspec = replace(
        runspec,
        input_location=location,
        payload=replace(
            runspec.payload,
            msa_set_manifest=manifest,
            msa_set=replace(
                runspec.payload.msa_set,
                artifact_set_id=manifest.artifact_set_id,
                member_a3m_paths=(first.logical_path, second.logical_path),
            ),
            fold_shard_projection=replace(
                runspec.payload.fold_shard_projection,
                sha256=hashlib.sha256(document).hexdigest(),
                size_bytes=len(document),
            ),
        ),
    )
    # Decompression is the existing test double; flatten still verifies and opens
    # these actual tiny tar bytes through the production pipeline.
    recording = _RecordingDeps(runspec, "bioir")
    return runspec, replace(recording.deps(), bioir_session_factory=deps.bioir_session_factory)
