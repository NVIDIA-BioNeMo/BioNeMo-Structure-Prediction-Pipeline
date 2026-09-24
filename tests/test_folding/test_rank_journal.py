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

"""Rank journal writer/reader tests.

The writer must append exactly one JSONL line per successful target event and
``fsync`` each event before returning; the bounded reader must round-trip the
full authority tuple, ignore a torn trailing append, and fail closed on a
malformed non-final line. The runtime qualification tuple id mirror is pinned
against the control-side formula to guard against drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bspp.orchestration.runtime.folding.rank_journal import (
    RankJournalEvent,
    RankJournalOutput,
    RankJournalWriter,
    folding_qualification_tuple_id,
    read_rank_journal,
)


def _sample_event(*, target_id: str = "AF-0000000000000001_AF-0000000000000002", rank: int = 0) -> RankJournalEvent:
    return RankJournalEvent(
        phase_run_id="phase-run-" + "a" * 32,
        attempt_id="attempt-0001",
        rank=rank,
        fold_action_id="fold-000001",
        fold_action_digest="b" * 64,
        shard_projection_sha256="c" * 64,
        shard_projection_worker_count=2,
        shard_projection_lpt_version=1,
        predecessor_digest="d" * 64,
        target_id=target_id,
        sequence_sha256="e" * 64,
        description=f"a3ms/{target_id}.a3m",
        backend="openfold-cli",
        qualification_tuple_id="f" * 64,
        outputs=(
            RankJournalOutput(path=f"/out/{target_id}.pdb", size=9, sha256="a" * 64),
            RankJournalOutput(path=f"/out/{target_id}.json", size=11, sha256="b" * 64),
        ),
    )


def test_writer_appends_one_jsonl_line_per_event_and_fsyncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fsync_calls: list[int] = []

    def recording_fsync(fd: int) -> None:
        fsync_calls.append(fd)

    monkeypatch.setattr("bspp.orchestration.runtime.folding.rank_journal.os.fsync", recording_fsync)

    path = tmp_path / "ranks" / "0" / "journal.jsonl"
    first = _sample_event(target_id="t1")
    second = _sample_event(target_id="t2")
    with RankJournalWriter(path) as journal:
        # Construction fsyncs the parent directory so the journal entry is durable.
        assert len(fsync_calls) == 1
        journal.append(first)
        assert len(fsync_calls) == 2
        journal.append(second)
        assert len(fsync_calls) == 3

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [first.to_json_line().rstrip("\n"), second.to_json_line().rstrip("\n")]


def test_writer_creates_empty_file_for_empty_rank(tmp_path: Path) -> None:
    path = tmp_path / "ranks" / "3" / "journal.jsonl"
    with RankJournalWriter(path):
        pass
    assert path.read_bytes() == b""


def test_read_rank_journal_round_trips_and_ignores_torn_final_append(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    event = _sample_event()
    with RankJournalWriter(path) as journal:
        journal.append(event)

    # A torn trailing append: a partial JSON line with no trailing newline.
    with path.open("ab") as handle:
        handle.write(b'{"schema_version": 1, "phase_run_id": "trunc')

    events = read_rank_journal(path)
    assert events == (event,)


def test_read_rank_journal_malformed_non_final_line_raises(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_bytes(b'{"schema_version": 1, "phase_run_id": "trunc\n' + _sample_event().to_json_line().encode("utf-8"))
    with pytest.raises(ValueError, match="malformed rank journal line"):
        read_rank_journal(path)


def test_full_authority_tuple_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    event = _sample_event()
    with RankJournalWriter(path) as journal:
        journal.append(event)
    assert read_rank_journal(path) == (event,)


def test_folding_qualification_tuple_id_pinned() -> None:
    assert (
        folding_qualification_tuple_id(
            backend="openfold-cli",
            kernel_image="kernel.sqsh",
            cluster_snapshot_digest="b" * 64,
        )
        == "a9f1270934a2f2e5bd98e25a719794e991f2abd0a77b0e6675e89726c86f7bc1"
    )
    assert (
        folding_qualification_tuple_id(
            backend="colabfold",
            kernel_image="colabfold.sqsh",
            cluster_snapshot_digest="c" * 64,
        )
        == "578b948917d9d0ea8ef5e39f0cbe082253ba98e2c1976a66c9a68e49deb9f202"
    )


def test_rank_journal_event_loader_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    event = _sample_event()
    mapping = event.to_mapping()
    mapping["surprise"] = "nope"
    path.write_text(json.dumps(mapping) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown RankJournalEvent field"):
        read_rank_journal(path)
