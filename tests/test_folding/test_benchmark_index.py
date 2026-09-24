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

"""Tests for the canonical-pair index schema, strict loader, and builder API."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from bspp.orchestration.contract.runspec import VALID_TOOL_USED
from bspp.orchestration.runtime.folding.benchmark.index import (
    CanonicalPairIndex,
    CanonicalPairIndexEntry,
    build_canonical_pair_index,
    load_canonical_pair_index,
    write_canonical_pair_index,
)

_VALID_TOOL = "OpenFold / AlphaFold-Multimer"


def _entry(
    target_id: str = "pdb-temporal-2022-2025-v1/0001",
    sequence_sha256: str = "a" * 64,
    model_entity_id: str = "AF-0000000000000001",
    tool_used: str = _VALID_TOOL,
    structure_path: str = "AF-0000000000000001-model_v1.pdb",
    scores_path: str = "AF-0000000000000001-meta_v1.json",
) -> CanonicalPairIndexEntry:
    return CanonicalPairIndexEntry(
        target_id=target_id,
        sequence_sha256=sequence_sha256,
        model_entity_id=model_entity_id,
        tool_used=tool_used,
        structure_path=structure_path,
        scores_path=scores_path,
    )


def _entry_record(entry: CanonicalPairIndexEntry) -> tuple[str, str, str, str, str, str]:
    return (
        entry.target_id,
        entry.sequence_sha256,
        entry.model_entity_id,
        entry.tool_used,
        entry.structure_path,
        entry.scores_path,
    )


class TestCanonicalPairIndexEntrySchema:
    def test_valid_entry_round_trips_mapping(self) -> None:
        entry = _entry()
        assert entry.to_mapping() == {
            "target_id": "pdb-temporal-2022-2025-v1/0001",
            "sequence_sha256": "a" * 64,
            "model_entity_id": "AF-0000000000000001",
            "tool_used": _VALID_TOOL,
            "structure_path": "AF-0000000000000001-model_v1.pdb",
            "scores_path": "AF-0000000000000001-meta_v1.json",
        }

    def test_to_json_parses_back_to_mapping(self) -> None:
        entry = _entry()
        assert json.loads(entry.to_json()) == entry.to_mapping()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"target_id": ""}, "target_id"),
            ({"target_id": "   "}, "target_id"),
            ({"sequence_sha256": "a" * 63}, "64 lowercase hex"),
            ({"sequence_sha256": "g" * 64}, "64 lowercase hex"),
            ({"sequence_sha256": "A" * 64}, "64 lowercase hex"),
            ({"sequence_sha256": ""}, "64 lowercase hex"),
            ({"model_entity_id": ""}, "model_entity_id"),
            ({"tool_used": "not-a-tool"}, "tool_used"),
            ({"structure_path": ""}, "structure_path"),
            ({"scores_path": ""}, "scores_path"),
        ],
    )
    def test_invalid_fields_raise_value_error(self, kwargs: dict[str, str], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _entry(**kwargs)


class TestCanonicalPairIndexSchema:
    def test_valid_index_round_trips_mapping(self) -> None:
        entry = _entry()
        idx = CanonicalPairIndex(schema_version=1, run_id="run-1", entries=(entry,))
        assert idx.to_mapping() == {
            "schema_version": 1,
            "run_id": "run-1",
            "entries": [entry.to_mapping()],
        }

    @pytest.mark.parametrize(
        ("schema_version", "run_id", "entries", "match"),
        [
            (2, "run-1", (_entry(),), "schema_version"),
            (0, "run-1", (_entry(),), "schema_version"),
            (1, "", (_entry(),), "run_id"),
        ],
    )
    def test_invalid_index_fields_raise_value_error(
        self,
        schema_version: int,
        run_id: str,
        entries: list[CanonicalPairIndexEntry],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            CanonicalPairIndex(schema_version=schema_version, run_id=run_id, entries=tuple(entries))

    def test_duplicate_target_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate target_id"):
            CanonicalPairIndex(
                schema_version=1,
                run_id="run-1",
                entries=(_entry(), _entry(model_entity_id="AF-0000000000000002")),
            )

    def test_empty_entries_is_allowed(self) -> None:
        idx = CanonicalPairIndex(schema_version=1, run_id="run-1", entries=())
        assert idx.entries == ()

    def test_non_tuple_entries_rejected(self) -> None:
        with pytest.raises(ValueError, match="entries"):
            CanonicalPairIndex(schema_version=1, run_id="run-1", entries=[_entry()])  # type: ignore[arg-type]


class TestLoadCanonicalPairIndex:
    def test_absent_file_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Cannot read"):
            load_canonical_pair_index(tmp_path / "missing.json")

    def test_malformed_json_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_canonical_pair_index(path)

    def test_non_object_top_level_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1,2,3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load_canonical_pair_index(path)

    def test_missing_schema_version_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        path.write_text(json.dumps({"run_id": "run-1", "entries": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="schema_version"):
            load_canonical_pair_index(path)

    def test_unknown_schema_version_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        path.write_text(
            json.dumps({"schema_version": 2, "run_id": "run-1", "entries": []}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="schema_version"):
            load_canonical_pair_index(path)

    def test_duplicate_target_id_in_json_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        payload = {
            "schema_version": 1,
            "run_id": "run-1",
            "entries": [
                _entry().to_mapping(),
                _entry(model_entity_id="AF-0000000000000002").to_mapping(),
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate target_id"):
            load_canonical_pair_index(path)

    def test_invalid_entry_field_in_json_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "index.json"
        entry = _entry().to_mapping()
        entry["sequence_sha256"] = "g" * 64
        payload = {"schema_version": 1, "run_id": "run-1", "entries": [entry]}
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="64 lowercase hex"):
            load_canonical_pair_index(path)


class TestBuilder:
    def test_builder_orders_entries_by_target_id(self) -> None:
        idx = build_canonical_pair_index(
            run_id="r",
            entries=[
                _entry_record(_entry(target_id="t2", model_entity_id="AF-0000000000000002")),
                _entry_record(_entry(target_id="t1")),
            ],
        )
        assert [entry.target_id for entry in idx.entries] == ["t1", "t2"]

    def test_builder_is_deterministic(self) -> None:
        entries = [
            _entry_record(_entry(target_id="t2", model_entity_id="AF-0000000000000002")),
            _entry_record(_entry(target_id="t1")),
        ]
        first = build_canonical_pair_index(run_id="r", entries=entries)
        second = build_canonical_pair_index(run_id="r", entries=entries)
        assert first.to_mapping() == second.to_mapping()
        assert first.to_json() == second.to_json()

    def test_builder_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="Duplicate target_id"):
            build_canonical_pair_index(
                run_id="r",
                entries=[
                    _entry_record(_entry()),
                    _entry_record(_entry(model_entity_id="AF-0000000000000002")),
                ],
            )


class TestWriteAndReload:
    def test_write_reload_round_trip(self, tmp_path: Path) -> None:
        idx = build_canonical_pair_index(
            run_id="run-1",
            entries=[
                _entry_record(_entry(target_id="t2", model_entity_id="AF-0000000000000002")),
                _entry_record(_entry(target_id="t1")),
            ],
        )
        path = tmp_path / "nested" / "index.json"
        write_canonical_pair_index(idx, path)
        loaded = load_canonical_pair_index(path)
        assert loaded == idx
        assert loaded.to_mapping() == idx.to_mapping()

    def test_write_is_deterministic_bytes(self, tmp_path: Path) -> None:
        idx = build_canonical_pair_index(run_id="r", entries=[_entry_record(_entry())])
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        write_canonical_pair_index(idx, first)
        write_canonical_pair_index(idx, second)
        assert first.read_bytes() == second.read_bytes()

    def test_write_leaves_no_temp_file(self, tmp_path: Path) -> None:
        idx = build_canonical_pair_index(run_id="r", entries=[_entry_record(_entry())])
        path = tmp_path / "index.json"
        write_canonical_pair_index(idx, path)
        assert not (tmp_path / "index.json.tmp").exists()


class TestFrozenImmutability:
    def test_entry_is_frozen(self) -> None:
        entry = _entry()
        with pytest.raises(FrozenInstanceError):
            entry.target_id = "other"  # type: ignore[misc]

    def test_index_is_frozen(self) -> None:
        idx = CanonicalPairIndex(schema_version=1, run_id="run-1", entries=(_entry(),))
        with pytest.raises(FrozenInstanceError):
            idx.entries = ()  # type: ignore[misc]


def test_valid_tool_used_contains_expected_strings() -> None:
    # The index gates tool_used against the frozen contract tuple; pin the two
    # strings the task verifies rely on so a contract drift is caught here.
    assert "OpenFold / AlphaFold-Multimer" in VALID_TOOL_USED
    assert "OpenFold-TRT / AlphaFold-Multimer" in VALID_TOOL_USED
