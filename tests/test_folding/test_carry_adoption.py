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

"""Unit tests for verified-copy folding carry adoption."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_carry_forward import (
    FoldingCarryForwardContent,
    FoldingCarryForwardOutput,
    FoldingCarryForwardRecord,
    folding_carry_forward_id,
)
from bspp.orchestration.contract.phase import canonical_mapping_digest
from bspp.orchestration.contract.versioning import CURRENT_CONTRACT_SCHEMA_VERSION
from bspp.orchestration.runtime.folding.carry_adoption import (
    CarryAdoptionAuthorityBinding,
    CarryAdoptionError,
    adopt_all_carried_outputs,
    adopt_carried_outputs,
    copy_and_verify,
    read_adopted_journal,
)

_PHASE_RUN_ID = "phase-run-0123456789abcdef0123456789abcdef"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _output(output_path: str, data: bytes) -> FoldingCarryForwardOutput:
    return FoldingCarryForwardOutput(output_path=output_path, size_bytes=len(data), sha256=_sha256(data))


def _content(
    target_id: str,
    source_rank: int,
    outputs: tuple[FoldingCarryForwardOutput, ...],
) -> FoldingCarryForwardContent:
    return FoldingCarryForwardContent(
        target_id=target_id,
        sequence_sha256="e" * 64,
        source_action_id="fold-000001",
        source_rank=source_rank,
        outputs=outputs,
    )


def _record(
    content: tuple[FoldingCarryForwardContent, ...],
    *,
    source_attempt_id: str = "attempt-0001",
    target_attempt_id: str = "attempt-0002",
) -> FoldingCarryForwardRecord:
    content_digest = canonical_mapping_digest(
        {"schema_version": CURRENT_CONTRACT_SCHEMA_VERSION, "content": [item.to_mapping() for item in content]}
    )
    identity: dict[str, object] = {
        "schema_version": CURRENT_CONTRACT_SCHEMA_VERSION,
        "phase_run_id": _PHASE_RUN_ID,
        "phase_plan_digest": "a" * 64,
        "source_attempt_id": source_attempt_id,
        "source_attempt_ordinal": int(source_attempt_id.rsplit("-", maxsplit=1)[1]),
        "source_runspec_digest": "b" * 64,
        "target_attempt_id": target_attempt_id,
        "target_attempt_ordinal": int(target_attempt_id.rsplit("-", maxsplit=1)[1]),
        "backend": "openfold-cli",
        "content": [item.to_mapping() for item in content],
        "ancestor_closure": [],
        "content_digest": content_digest,
        "declared_at": "2026-09-11T12:00:00Z",
    }
    return FoldingCarryForwardRecord(
        folding_carry_forward_id=folding_carry_forward_id(identity),
        phase_run_id=_PHASE_RUN_ID,
        phase_plan_digest="a" * 64,
        source_attempt_id=source_attempt_id,
        source_attempt_ordinal=int(source_attempt_id.rsplit("-", maxsplit=1)[1]),
        source_runspec_digest="b" * 64,
        target_attempt_id=target_attempt_id,
        target_attempt_ordinal=int(target_attempt_id.rsplit("-", maxsplit=1)[1]),
        backend="openfold-cli",
        content=content,
        ancestor_closure=(),
        content_digest=content_digest,
        declared_at="2026-09-11T12:00:00Z",
    )


def _binding(worker_count: int, descriptions: dict[str, str]) -> CarryAdoptionAuthorityBinding:
    return CarryAdoptionAuthorityBinding(
        phase_run_id=_PHASE_RUN_ID,
        attempt_id="attempt-0002",
        fold_action_id="fold-000001",
        fold_action_digest="c" * 64,
        shard_projection_sha256="d" * 64,
        shard_projection_worker_count=worker_count,
        shard_projection_lpt_version=1,
        predecessor_digest="f" * 64,
        backend="openfold-cli",
        qualification_tuple_id="9" * 64,
        descriptions=descriptions,
    )


def test_copy_and_verify_copies_and_rechecks(tmp_path: Path) -> None:
    source = tmp_path / "source" / "structure.pdb"
    source.parent.mkdir(parents=True)
    data = b"bspp-carried-structure-bytes\n"
    source.write_bytes(data)
    source_before = source.read_bytes()
    target = tmp_path / "target" / "structure.pdb"

    copy_and_verify(source, target, size_bytes=len(data), sha256=_sha256(data))

    assert target.read_bytes() == data
    assert source.read_bytes() == source_before

    with pytest.raises(CarryAdoptionError, match="verification failed"):
        copy_and_verify(source, tmp_path / "bad-size", size_bytes=len(data) + 1, sha256=_sha256(data))
    with pytest.raises(CarryAdoptionError, match="verification failed"):
        copy_and_verify(source, tmp_path / "bad-hash", size_bytes=len(data), sha256="0" * 64)
    assert not (tmp_path / "bad-size").exists()
    assert not (tmp_path / "bad-hash").exists()


def test_copy_and_verify_leaves_no_partial_target_on_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.pdb"
    data = b"some-bytes\n"
    source.write_bytes(data)
    target = tmp_path / "targets" / "out.pdb"

    with pytest.raises(CarryAdoptionError):
        copy_and_verify(source, target, size_bytes=len(data), sha256="0" * 64)

    assert not target.exists()
    leftovers = [path for path in target.parent.iterdir() if path.name.startswith(".bspp-carry-")]
    assert leftovers == []


def test_adopt_carried_outputs_writes_adopted_events(tmp_path: Path) -> None:
    source_root = tmp_path / "predecessor"
    structure = source_root / "ranks" / "0" / "outputs" / "AF-0000000000000001" / "structure.pdb"
    scores = source_root / "ranks" / "0" / "outputs" / "AF-0000000000000001" / "scores.json"
    structure.parent.mkdir(parents=True)
    structure_data = b"structure\n"
    scores_data = b'{"plddt": []}\n'
    structure.write_bytes(structure_data)
    scores.write_bytes(scores_data)
    content = (
        _content(
            "AF-0000000000000001",
            0,
            (_output(str(structure), structure_data), _output(str(scores), scores_data)),
        ),
    )
    record = _record(content)
    action_root = tmp_path / "successor" / "actions" / "fold-000001"
    binding = _binding(2, {"AF-0000000000000001": "a3ms/AFDB_AF-0000000000000001.a3m"})

    adopted = adopt_carried_outputs(record, successor_action_root=action_root, rank=0, authority_binding=binding)

    assert adopted == ("AF-0000000000000001",)
    adopted_path = action_root / "ranks" / "0" / "adopted.jsonl"
    assert adopted_path.is_file()
    raw = adopted_path.read_bytes()
    assert raw.endswith(b"\n")
    payload = json.loads(raw.splitlines()[-1])
    assert payload["event_kind"] == "adopted"
    assert payload["source_attempt_id"] == "attempt-0001"
    assert payload["carry_record_digest"] == record.digest
    target_structure = action_root / "ranks" / "0" / "outputs" / "AF-0000000000000001" / "structure.pdb"
    target_scores = action_root / "ranks" / "0" / "outputs" / "AF-0000000000000001" / "scores.json"
    assert target_structure.read_bytes() == structure_data
    assert target_scores.read_bytes() == scores_data
    assert structure.read_bytes() == structure_data

    events = read_adopted_journal(adopted_path)
    assert [event.target_id for event in events] == ["AF-0000000000000001"]
    assert events[0].description == "a3ms/AFDB_AF-0000000000000001.a3m"


def test_adopt_all_carried_outputs_produces_complete_closure(tmp_path: Path) -> None:
    source_root = tmp_path / "predecessor"
    outputs_by_rank: dict[int, dict[str, tuple[bytes, bytes]]] = {}
    content_items: list[FoldingCarryForwardContent] = []
    for rank, target_id in ((0, "AF-0000000000000001"), (1, "AF-0000000000000002")):
        structure = source_root / "ranks" / str(rank) / "outputs" / target_id / "structure.pdb"
        scores = source_root / "ranks" / str(rank) / "outputs" / target_id / "scores.json"
        structure.parent.mkdir(parents=True)
        structure_data = f"{target_id}-structure\n".encode()
        scores_data = f"{target_id}-scores\n".encode()
        structure.write_bytes(structure_data)
        scores.write_bytes(scores_data)
        outputs_by_rank[rank] = {target_id: (structure_data, scores_data)}
        content_items.append(
            _content(
                target_id,
                rank,
                (_output(str(structure), structure_data), _output(str(scores), scores_data)),
            )
        )
    record = _record(tuple(content_items))
    action_root = tmp_path / "successor" / "actions" / "fold-000001"
    descriptions = {
        "AF-0000000000000001": "a3ms/AFDB_AF-0000000000000001.a3m",
        "AF-0000000000000002": "a3ms/AFDB_AF-0000000000000002.a3m",
    }
    binding = _binding(2, descriptions)

    adopted = adopt_all_carried_outputs(record, successor_action_root=action_root, authority_binding=binding)

    assert sorted(adopted) == ["AF-0000000000000001", "AF-0000000000000002"]
    for rank, target_id in ((0, "AF-0000000000000001"), (1, "AF-0000000000000002")):
        adopted_path = action_root / "ranks" / str(rank) / "adopted.jsonl"
        events = read_adopted_journal(adopted_path)
        assert [event.target_id for event in events] == [target_id]
        structure_data, scores_data = outputs_by_rank[rank][target_id]
        target_structure = action_root / "ranks" / str(rank) / "outputs" / target_id / "structure.pdb"
        target_scores = action_root / "ranks" / str(rank) / "outputs" / target_id / "scores.json"
        assert target_structure.read_bytes() == structure_data
        assert target_scores.read_bytes() == scores_data
