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

"""Focused tests for folding checkpoint merge and resume seams."""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from bspp.orchestration.contract.folding_checkpoint import (
    FoldingCheckpointState,
    FoldingRemainingWorkPlan,
    folding_checkpoint_state_from_mapping,
    folding_completion_record_from_mapping,
    folding_failure_record_from_mapping,
    folding_remaining_work_plan_from_mapping,
)
from bspp.orchestration.contract.versioning import UnsupportedSchemaVersionError
from bspp.orchestration.runtime.folding.checkpoint import (
    load_checkpoint_state,
    merge_checkpoint_records,
    write_merged_checkpoint_views,
)
from bspp.orchestration.runtime.folding.resume import classify_failure, plan_checkpoint_resume


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _completion_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "protein_id": "model-a",
        "runtime_seconds": 12.5,
        "timestamp": "2026-02-08T12:34:56",
        "node_id": "node-0",
        "gpu_id": 0,
        "source_layout": "current-per-node",
        "source_path": "/checkpoints/completed_node_0.csv",
        "source_ordinal": 0,
        "row_number": 2,
    }
    payload.update(overrides)
    return payload


def _failure_mapping(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "protein_id": "model-b",
        "error_message": "TIMEOUT after 600 seconds",
        "timestamp": "2026-02-08T12:35:28",
        "node_id": None,
        "gpu_id": None,
        "source_layout": "legacy-per-gpu",
        "source_path": "/checkpoints/failed_gpu_1.csv",
        "source_ordinal": 1,
        "row_number": 3,
    }
    payload.update(overrides)
    return payload


def test_checkpoint_contract_records_are_versioned_immutable_and_fail_closed() -> None:
    completion = folding_completion_record_from_mapping(_completion_mapping())
    failure = folding_failure_record_from_mapping(_failure_mapping())
    state = FoldingCheckpointState(completions=(completion,), failures=(failure,))

    assert folding_checkpoint_state_from_mapping(state.to_mapping()) == state
    assert completion.to_mapping() == _completion_mapping()
    assert failure.to_mapping() == _failure_mapping()

    with pytest.raises(FrozenInstanceError):
        completion.protein_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        state.completions = ()  # type: ignore[misc]
    with pytest.raises(UnsupportedSchemaVersionError):
        folding_completion_record_from_mapping(_completion_mapping(schema_version=2))
    with pytest.raises(ValueError, match="unknown fields"):
        folding_failure_record_from_mapping(_failure_mapping(extra="no"))
    with pytest.raises(ValueError, match="runtime_seconds"):
        folding_completion_record_from_mapping(_completion_mapping(runtime_seconds=float("nan")))
    with pytest.raises(ValueError, match="legacy-per-gpu"):
        folding_failure_record_from_mapping(_failure_mapping(node_id="unexpected"))
    with pytest.raises(ValueError, match="strictly sorted"):
        FoldingCheckpointState(
            completions=(
                folding_completion_record_from_mapping(_completion_mapping(protein_id="model-z")),
                completion,
            ),
            failures=(failure,),
        )
    with pytest.raises(ValueError, match="both completed and failed"):
        FoldingCheckpointState(
            completions=(completion,),
            failures=(folding_failure_record_from_mapping(_failure_mapping(protein_id="model-a")),),
        )

    remaining = plan_checkpoint_resume(("model-a", "model-b"), checkpoint_state=state)
    for record in (completion, failure, state, remaining):
        with pytest.raises(ValueError, match="schema_version must be declared explicitly"):
            replace(record, schema_version=None)


def test_load_checkpoint_state_normalizes_current_and_legacy_with_stable_conflict_resolution(
    tmp_path: Path,
) -> None:
    current_completed = _write_csv(
        tmp_path / "completed_node_9.csv",
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\n"
        "model-a,10.5,2026-02-08T10:00:00,node-9,72\n"
        "model-c,11,2026-02-08T10:01:00,node-9,73\n",
    )
    legacy_completed = _write_csv(
        tmp_path / "completed_gpu_1.csv",
        "protein_id,runtime_seconds,timestamp\nmodel-a,20.25,2026-02-08T11:00:00\n",
    )
    current_failed = _write_csv(
        tmp_path / "failed_node_9.csv",
        "protein_id,error_message,timestamp,node_id,gpu_id\n"
        "model-a,OOM,2026-02-08T09:00:00,node-9,72\n"
        "model-b,TIMEOUT,2026-02-08T09:01:00,node-9,73\n"
        "model-d,OOM,2026-02-08T09:02:00,node-9,74\n",
    )
    earlier_current_failed = _write_csv(
        tmp_path / "failed_node_1.csv",
        "protein_id,error_message,timestamp,node_id,gpu_id\nmodel-d,TIMEOUT,2026-02-08T08:00:00,node-1,8\n",
    )
    legacy_failed = _write_csv(
        tmp_path / "failed_gpu_1.csv",
        "protein_id,error_message,timestamp\nmodel-b,EXIT_CODE 1,2026-02-08T12:00:00\n",
    )

    state = load_checkpoint_state(
        current_node_completion_shards=(current_completed,),
        legacy_completion_shards=(legacy_completed,),
        current_node_failure_shards=(current_failed, earlier_current_failed),
        legacy_failure_shards=(legacy_failed,),
    )

    assert tuple(record.protein_id for record in state.completions) == ("model-a", "model-c")
    assert state.completions[0].runtime_seconds == 20.25
    assert state.completions[0].source_layout == "legacy-per-gpu"
    assert state.completions[0].node_id is None
    assert state.completions[0].gpu_id is None
    assert tuple(record.protein_id for record in state.failures) == ("model-b", "model-d")
    assert state.failures[0].error_message == "EXIT_CODE 1"
    assert state.failures[0].source_layout == "legacy-per-gpu"
    assert state.failures[1].error_message == "OOM"
    assert state.failures[1].source_path == str(current_failed)


def test_current_gpu_shards_follow_node_shards_regardless_of_schema(tmp_path: Path) -> None:
    node_failure = _write_csv(
        tmp_path / "failed_node_3.csv",
        "protein_id,error_message,timestamp,node_id,gpu_id\nmodel-a,OOM,2026-02-08T10:00:00,node-3,24\n",
    )
    gpu_failure = _write_csv(
        tmp_path / "failed_gpu_0.csv",
        "protein_id,error_message,timestamp,node_id,gpu_id\nmodel-a,TIMEOUT,2026-02-08T11:00:00,node-0,0\n",
    )

    state = load_checkpoint_state(
        current_node_failure_shards=(node_failure,),
        current_gpu_failure_shards=(gpu_failure,),
    )

    assert state.failures[0].error_message == "TIMEOUT"
    assert state.failures[0].source_layout == "current-per-gpu"
    assert plan_checkpoint_resume(("model-a",), checkpoint_state=state).remaining_model_ids == ("model-a",)


def test_load_checkpoint_state_fails_closed_on_malformed_headers_rows_and_values(tmp_path: Path) -> None:
    bad_header = _write_csv(
        tmp_path / "bad-header.csv",
        "protein_id,runtime_seconds,timestamp,gpu_id,node_id\nmodel-a,10,2026-02-08T10:00:00,0,node-0\n",
    )
    with pytest.raises(ValueError, match="header"):
        load_checkpoint_state(current_node_completion_shards=(bad_header,))

    short_row = _write_csv(
        tmp_path / "short-row.csv",
        "protein_id,error_message,timestamp,node_id,gpu_id\nmodel-a,OOM,2026-02-08T10:00:00,node-0\n",
    )
    with pytest.raises(ValueError, match="expected 5"):
        load_checkpoint_state(current_node_failure_shards=(short_row,))

    invalid_value = _write_csv(
        tmp_path / "invalid-value.csv",
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\nmodel-a,nan,2026-02-08T10:00:00,node-0,0\n",
    )
    with pytest.raises(ValueError, match="runtime_seconds"):
        load_checkpoint_state(current_node_completion_shards=(invalid_value,))

    blank_identity = _write_csv(
        tmp_path / "blank-identity.csv",
        "protein_id,error_message,timestamp\n,TIMEOUT,2026-02-08T10:00:00\n",
    )
    with pytest.raises(ValueError, match="protein_id"):
        load_checkpoint_state(legacy_failure_shards=(blank_identity,))

    with pytest.raises(ValueError, match="declared more than once"):
        load_checkpoint_state(
            current_node_completion_shards=(invalid_value,),
            legacy_completion_shards=(invalid_value,),
        )


def test_load_checkpoint_state_tolerates_only_a_torn_final_append(tmp_path: Path) -> None:
    completion = _write_csv(
        tmp_path / "completed_node_0.csv",
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\n"
        "model-a,10,2026-02-08T10:00:00,node-0,0\n"
        "model-torn,11,2026-02-08T10:01",
    )

    state = load_checkpoint_state(current_node_completion_shards=(completion,))

    assert tuple(record.protein_id for record in state.completions) == ("model-a",)


def test_load_checkpoint_state_accepts_empty_and_blank_baseline_rows(tmp_path: Path) -> None:
    empty = _write_csv(tmp_path / "completed_node_empty.csv", "")
    newline_only = _write_csv(tmp_path / "completed_node_newline.csv", "\n")
    blank = _write_csv(
        tmp_path / "completed_node_blank.csv",
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\n\nmodel-a,10,2026-02-08T10:00:00,node-0,0\n\n",
    )

    state = load_checkpoint_state(current_node_completion_shards=(empty, newline_only, blank))

    assert tuple(record.protein_id for record in state.completions) == ("model-a",)


@pytest.mark.parametrize(
    ("raw_message", "expected_message"),
    [
        ("", ""),
        ('"worker" not found', "worker not found"),
        ("TIMEOUT after worker stop ", "TIMEOUT after worker stop"),
        ("CUDA error\rfrom subprocess", "CUDA error from subprocess"),
    ],
)
def test_load_checkpoint_state_accepts_baseline_sanitized_error_messages(
    tmp_path: Path,
    raw_message: str,
    expected_message: str,
) -> None:
    failure = _write_csv(
        tmp_path / "failed_gpu_0.csv",
        f"protein_id,error_message,timestamp\n model-a ,{raw_message},2026-02-08T10:00:00\n",
    )

    state = load_checkpoint_state(legacy_failure_shards=(failure,))

    assert state.failures[0].protein_id == "model-a"
    assert state.failures[0].error_message == expected_message
    assert classify_failure(expected_message) == ("transient" if "TIMEOUT" in expected_message else "permanent")


def test_load_checkpoint_state_normalizes_baseline_af_identifiers(tmp_path: Path) -> None:
    shard = _write_csv(
        tmp_path / "completed_node_0.csv",
        "protein_id,runtime_seconds,timestamp,node_id,gpu_id\nAF_000_AF_111.merged,10,2026-02-08T10:00:00,node-0,0\n",
    )

    state = load_checkpoint_state(current_node_completion_shards=(shard,))

    assert state.completions[0].protein_id == "AF-000_AF-111.merged"
    plan = plan_checkpoint_resume(("AF_000_AF_111.merged",), checkpoint_state=state)
    assert plan.completed_model_ids == ("AF_000_AF_111.merged",)
    assert plan.remaining_model_ids == ()


def test_merge_checkpoint_records_rejects_ambiguous_equal_precedence() -> None:
    first = folding_completion_record_from_mapping(_completion_mapping(source_path="/checkpoints/completed_node_1.csv"))
    second = folding_completion_record_from_mapping(
        _completion_mapping(
            runtime_seconds=99,
            source_path="/checkpoints/completed_node_2.csv",
        )
    )

    with pytest.raises(ValueError, match="precedence coordinate"):
        merge_checkpoint_records((first, second), ())


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("timeout after 600 seconds", "transient"),
        ("batch INCOMPLETE", "transient"),
        ("TIMEOUT followed by OOM", "transient"),
        ("OOM", "permanent"),
        ("OUT_OF_MEMORY", "permanent"),
        ("CUDA_OUT_OF_MEMORY", "permanent"),
        ("EXIT_CODE 1", "permanent"),
        ("unclassified worker error", "permanent"),
    ],
)
def test_classify_failure_preserves_baseline_precedence(message: str, expected: str) -> None:
    assert classify_failure(message) == expected


def test_plan_checkpoint_resume_preserves_declared_order_and_retry_failed_policy() -> None:
    state = FoldingCheckpointState(
        completions=(folding_completion_record_from_mapping(_completion_mapping(protein_id="model-a")),),
        failures=(
            folding_failure_record_from_mapping(
                _failure_mapping(protein_id="model-b", error_message="TIMEOUT", row_number=2)
            ),
            folding_failure_record_from_mapping(
                _failure_mapping(protein_id="model-c", error_message="CUDA_OUT_OF_MEMORY", row_number=3)
            ),
            folding_failure_record_from_mapping(
                _failure_mapping(protein_id="model-extra", error_message="EXIT_CODE 2", row_number=4)
            ),
        ),
    )
    planned = ("model-d", "model-a", "model-c", "model-b")

    default_plan = plan_checkpoint_resume(planned, checkpoint_state=state)
    retry_plan = plan_checkpoint_resume(planned, checkpoint_state=state, retry_failed=True)

    assert default_plan == FoldingRemainingWorkPlan(
        planned_model_ids=planned,
        completed_model_ids=("model-a",),
        transient_failure_ids=("model-b",),
        permanent_failure_ids=("model-c",),
        remaining_model_ids=("model-d", "model-b"),
        unplanned_checkpoint_ids=("model-extra",),
        retry_failed=False,
    )
    assert retry_plan.remaining_model_ids == ("model-d", "model-c", "model-b")
    assert retry_plan.retry_failed is True
    assert folding_remaining_work_plan_from_mapping(default_plan.to_mapping()) == default_plan
    with pytest.raises(UnsupportedSchemaVersionError):
        folding_remaining_work_plan_from_mapping({**default_plan.to_mapping(), "schema_version": 2})
    with pytest.raises(ValueError, match="unknown fields"):
        folding_remaining_work_plan_from_mapping({**default_plan.to_mapping(), "extra": "no"})
    with pytest.raises(ValueError, match="remaining_model_ids"):
        folding_remaining_work_plan_from_mapping({**default_plan.to_mapping(), "remaining_model_ids": []})
    with pytest.raises(FrozenInstanceError):
        default_plan.remaining_model_ids = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="duplicate planned model identity"):
        plan_checkpoint_resume(("model-a", "model-a"), checkpoint_state=state)


def test_plan_checkpoint_resume_returns_empty_work_for_fully_completed_set() -> None:
    planned = ("model-c", "model-a", "model-b")
    completions = tuple(
        folding_completion_record_from_mapping(
            _completion_mapping(
                protein_id=protein_id,
                source_ordinal=source_ordinal,
                row_number=source_ordinal + 2,
            )
        )
        for source_ordinal, protein_id in enumerate(sorted(planned))
    )
    state = FoldingCheckpointState(completions=completions, failures=())

    plan = plan_checkpoint_resume(planned, checkpoint_state=state)

    assert plan.completed_model_ids == planned
    assert plan.remaining_model_ids == ()
    assert load_checkpoint_state() == FoldingCheckpointState(completions=(), failures=())


def test_write_merged_checkpoint_views_is_deterministic_and_preserves_old_files_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FoldingCheckpointState(
        completions=(
            folding_completion_record_from_mapping(_completion_mapping(protein_id="model-a", runtime_seconds=10.0)),
        ),
        failures=(
            folding_failure_record_from_mapping(_failure_mapping(protein_id="model-b", error_message="TIMEOUT, retry")),
        ),
    )
    completed_path = tmp_path / "checkpoints" / "completed_all.csv"
    failed_path = tmp_path / "checkpoints" / "failed_all.csv"

    result = write_merged_checkpoint_views(
        state,
        completed_path=completed_path,
        failed_path=failed_path,
    )

    assert result == (completed_path, failed_path)
    assert completed_path.read_bytes() == (
        b"protein_id,runtime_seconds,timestamp,node_id,gpu_id\r\nmodel-a,10.0,2026-02-08T12:34:56,node-0,0\r\n"
    )
    assert failed_path.read_bytes() == (
        b'protein_id,error_message,timestamp,node_id,gpu_id\r\nmodel-b,"TIMEOUT, retry",2026-02-08T12:35:28,,\r\n'
    )
    first_bytes = (completed_path.read_bytes(), failed_path.read_bytes())
    write_merged_checkpoint_views(state, completed_path=completed_path, failed_path=failed_path)
    assert (completed_path.read_bytes(), failed_path.read_bytes()) == first_bytes

    completed_path.write_bytes(b"old completed\n")
    failed_path.write_bytes(b"old failed\n")

    def injected_replace_failure(_: Path, __: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", injected_replace_failure)
    with pytest.raises(OSError, match="injected replace failure"):
        write_merged_checkpoint_views(state, completed_path=completed_path, failed_path=failed_path)

    assert completed_path.read_bytes() == b"old completed\n"
    assert failed_path.read_bytes() == b"old failed\n"
    assert tuple((tmp_path / "checkpoints").glob(".*.tmp")) == ()


def test_write_merged_checkpoint_views_rolls_back_after_second_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FoldingCheckpointState(
        completions=(folding_completion_record_from_mapping(_completion_mapping()),),
        failures=(folding_failure_record_from_mapping(_failure_mapping()),),
    )
    completed_path = tmp_path / "completed_all.csv"
    failed_path = tmp_path / "failed_all.csv"
    completed_path.write_bytes(b"old completed\n")
    failed_path.write_bytes(b"old failed\n")
    real_replace = os.replace
    replace_calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected second replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="injected second replace failure"):
        write_merged_checkpoint_views(state, completed_path=completed_path, failed_path=failed_path)

    assert completed_path.read_bytes() == b"old completed\n"
    assert failed_path.read_bytes() == b"old failed\n"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_write_merged_checkpoint_views_preserves_backup_when_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = FoldingCheckpointState(
        completions=(folding_completion_record_from_mapping(_completion_mapping()),),
        failures=(folding_failure_record_from_mapping(_failure_mapping()),),
    )
    completed_path = tmp_path / "completed_all.csv"
    failed_path = tmp_path / "failed_all.csv"
    completed_path.write_bytes(b"old completed\n")
    failed_path.write_bytes(b"old failed\n")
    real_replace = os.replace
    replace_calls = 0

    def fail_second_replace_and_rollback(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected second replace failure")
        if replace_calls == 3:
            raise OSError("injected rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace_and_rollback)

    with pytest.raises(BaseExceptionGroup, match="recovery snapshots preserved") as captured:
        write_merged_checkpoint_views(state, completed_path=completed_path, failed_path=failed_path)

    assert [str(error) for error in captured.value.exceptions] == [
        "injected second replace failure",
        "injected rollback failure",
    ]
    recovery_paths = tuple(tmp_path.glob(".completed_all.csv.backup.*.tmp"))
    assert len(recovery_paths) == 1
    assert recovery_paths[0].read_bytes() == b"old completed\n"
    assert failed_path.read_bytes() == b"old failed\n"


def test_merged_checkpoint_views_with_legacy_rows_are_reloadable(tmp_path: Path) -> None:
    state = FoldingCheckpointState(
        completions=(
            folding_completion_record_from_mapping(
                _completion_mapping(
                    node_id=None,
                    gpu_id=None,
                    source_layout="legacy-per-gpu",
                )
            ),
        ),
        failures=(folding_failure_record_from_mapping(_failure_mapping()),),
    )
    completed_path = tmp_path / "completed_all.csv"
    failed_path = tmp_path / "failed_all.csv"

    write_merged_checkpoint_views(state, completed_path=completed_path, failed_path=failed_path)
    reloaded = load_checkpoint_state(
        current_node_completion_shards=(completed_path,),
        current_node_failure_shards=(failed_path,),
    )

    assert tuple(record.protein_id for record in reloaded.completions) == ("model-a",)
    assert reloaded.completions[0].source_layout == "legacy-per-gpu"
    assert tuple(record.protein_id for record in reloaded.failures) == ("model-b",)
    assert reloaded.failures[0].source_layout == "legacy-per-gpu"


def test_merged_checkpoint_runtime_seconds_round_trip_without_float_loss(tmp_path: Path) -> None:
    completion = folding_completion_record_from_mapping(_completion_mapping(runtime_seconds=0.30000000000000004))
    state = FoldingCheckpointState(completions=(completion,), failures=())
    completed_path = tmp_path / "completed_all.csv"
    failed_path = tmp_path / "failed_all.csv"

    write_merged_checkpoint_views(state, completed_path=completed_path, failed_path=failed_path)
    reloaded = load_checkpoint_state(current_node_completion_shards=(completed_path,))

    assert reloaded.completions[0].runtime_seconds == completion.runtime_seconds
