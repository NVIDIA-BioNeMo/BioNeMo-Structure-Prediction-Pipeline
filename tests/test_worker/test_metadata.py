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

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bspp.orchestration.runtime.worker import (
    MetadataCombineError,
    MetadataCombinePlan,
    check_meta_json_for_null,
    clean_metadata_json_outputs,
    clean_sub_shard_metadata_state,
    combine_metadata_files,
    finalize_metadata,
    metadata_destination_dir,
    plan_metadata_finalization,
    prefilter_batch_inputs,
)

REAL_TOOLKIT_COMBINE_METADATA = (
    Path(__file__).resolve().parents[3]
    / "bspp"
    / "AFDB-Integration-Kit"
    / "uniprot"
    / "scripts"
    / "combine_metadata.py"
)
REQUIRE_REAL_TOOLKIT_ENV_VAR = "BSPP_REQUIRE_REAL_TOOLKIT_COMBINE"


def test_clean_sub_shard_metadata_state_removes_json_dirs_and_pipeline_cache(tmp_path: Path) -> None:
    for dirname in ("model_jsons", "chain_jsons"):
        path = tmp_path / dirname
        path.mkdir()
        (path / "old.json").write_text("{}")
    cache = tmp_path / ".pipeline_cache.json"
    cache.write_text("{}")

    removed = clean_sub_shard_metadata_state(tmp_path)

    assert set(removed) == {tmp_path / "model_jsons", tmp_path / "chain_jsons", cache}
    assert not (tmp_path / "model_jsons").exists()
    assert not cache.exists()


def test_clean_metadata_json_outputs_removes_failed_model_metadata_only(tmp_path: Path) -> None:
    for dirname in ("model_jsons", "chain_jsons"):
        path = tmp_path / dirname
        path.mkdir()
        (path / "AF-1.json").write_text("{}")
        (path / "AF-2.json").write_text("{}")

    assert clean_metadata_json_outputs(tmp_path, ["AF-1"]) == 2
    assert not (tmp_path / "model_jsons" / "AF-1.json").exists()
    assert (tmp_path / "model_jsons" / "AF-2.json").exists()


def test_check_meta_json_for_null_fast_path_matches_legacy_strings(tmp_path: Path) -> None:
    missing = tmp_path / "missing-meta_v1.json"
    assert check_meta_json_for_null(missing) == (False, "")

    path = tmp_path / "AF-1-meta_v1.json"
    path.write_text('{"pae":[1,null,2],"plddt":[1],"max_pae":[2]}')
    assert check_meta_json_for_null(path) == (True, "null in input json")


def test_check_meta_json_for_null_slow_path_reports_parse_and_schema_errors(tmp_path: Path) -> None:
    path = tmp_path / "AF-1-meta_v1.json"
    path.write_text("{")
    has_null, reason = check_meta_json_for_null(path, fast=False)
    assert has_null is True
    assert reason.startswith("JSON parse error:")

    path.write_text('{"pae":[1],"plddt":[1]}')
    assert check_meta_json_for_null(path, fast=False) == (True, "missing key: max_pae")


def test_prefilter_batch_inputs_preserves_order_and_uses_file_index(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "custom-good-meta_v1.json").write_text('{"pae":[1],"plddt":[1],"max_pae":[2]}')
    (input_dir / "AF-0000000000000002-meta_v1.json").write_text('{"pae":[null],"plddt":[1],"max_pae":[2]}')

    result = prefilter_batch_inputs(
        ["AF-0000000000000001", "AF-0000000000000002", "AF-0000000000000003"],
        input_dir,
        file_index={"AF-0000000000000001": ["custom-good-meta_v1.json"]},
    )

    assert result.good_model_ids == ("AF-0000000000000001", "AF-0000000000000003")
    assert result.bad_models == (("AF-0000000000000002", "null in input json"),)


def test_metadata_destination_dir_matches_upload_mode(tmp_path: Path) -> None:
    assert (
        metadata_destination_dir(tmp_path / "work", tmp_path / "shard" / "success_outputs", s3_upload_enabled=True)
        == tmp_path / "work" / "_metadata_staging"
    )
    assert (
        metadata_destination_dir(tmp_path / "work", tmp_path / "shard" / "success_outputs", s3_upload_enabled=False)
        == tmp_path / "shard" / "success_outputs"
    )


def test_plan_and_finalize_metadata_uses_legacy_names_and_injected_runner(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    (work_dir / "model_jsons").mkdir(parents=True)
    (work_dir / "chain_jsons").mkdir()
    output_dir = tmp_path / "success_outputs"
    commands: list[MetadataCombinePlan] = []

    plan = plan_metadata_finalization(
        work_dir=work_dir,
        output_base_dir=output_dir,
        logical_shard_id=7,
        total_shards=9,
        dataset_tag="wp8d",
    )

    assert [combine.output_filename for combine in plan.plans] == [
        "AF-metadata-8-of-9-wp8d.json",
        "AF-chain-metadata-8-of-9-wp8d.json",
    ]

    def runner(combine_plan: MetadataCombinePlan) -> int:
        commands.append(combine_plan)
        combine_plan.output_dir.mkdir(parents=True, exist_ok=True)
        (combine_plan.output_dir / combine_plan.output_filename).write_text("{}")
        return 0

    result = finalize_metadata(plan, runner=runner)

    assert result.exit_code == 0
    assert result.output_files == (
        output_dir / "metadata" / "search" / "AF-metadata-8-of-9-wp8d.json",
        output_dir / "metadata" / "collection" / "AF-chain-metadata-8-of-9-wp8d.json",
    )
    assert commands[0].input_dir == work_dir / "model_jsons"


def test_finalize_metadata_writes_real_shape_content_and_trailing_newline(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    _write_json(work_dir / "model_jsons" / "AF-0002.json", {"model_id": "AF-0002", "created": "2026-05-07T00:00:00Z"})
    _write_json(work_dir / "model_jsons" / "AF-0001.json", {"model_id": "AF-0001", "created": "2026-05-07T00:00:00Z"})
    _write_json(work_dir / "chain_jsons" / "AF-0002.json", {"model_id": "AF-0002", "chain": "B"})
    _write_json(work_dir / "chain_jsons" / "AF-0001.json", {"model_id": "AF-0001", "chain": "A"})

    output_dir = tmp_path / "success_outputs"
    plan = plan_metadata_finalization(
        work_dir=work_dir,
        output_base_dir=output_dir,
        logical_shard_id=3,
        total_shards=5,
        dataset_tag="wpn7",
    )

    result = finalize_metadata(plan)

    assert result.exit_code == 0
    assert result.output_files == (
        output_dir / "metadata" / "search" / "AF-metadata-4-of-5-wpn7.json",
        output_dir / "metadata" / "collection" / "AF-chain-metadata-4-of-5-wpn7.json",
    )
    search_rows = json.loads(result.output_files[0].read_text())
    collection_rows = json.loads(result.output_files[1].read_text())
    assert isinstance(search_rows, list)
    assert isinstance(collection_rows, list)
    assert search_rows == [
        {"created": "2026-05-07T00:00:00Z", "model_id": "AF-0001"},
        {"created": "2026-05-07T00:00:00Z", "model_id": "AF-0002"},
    ]
    assert collection_rows == [
        {"chain": "A", "model_id": "AF-0001"},
        {"chain": "B", "model_id": "AF-0002"},
    ]
    assert result.output_files[0].read_bytes().endswith(b"\n")


def test_combine_metadata_files_sorts_files_filters_directories_and_flattens_records(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    (input_dir / "AF-0000.json").mkdir(parents=True)
    _write_json(
        input_dir / "AF-0002.json",
        [{"model_id": "AF-0002", "chain": "A"}, {"model_id": "AF-0002", "chain": "B"}],
    )
    _write_json(input_dir / "AF-0001.json", {"model_id": "AF-0001", "chain": "A"})

    outputs = combine_metadata_files(input_dir=input_dir, output_dir=output_dir, output_filename="combined.json")

    assert outputs == ((output_dir / "combined.json").resolve(),)
    assert json.loads((output_dir / "combined.json").read_text()) == [
        {"chain": "A", "model_id": "AF-0001"},
        {"chain": "A", "model_id": "AF-0002"},
        {"chain": "B", "model_id": "AF-0002"},
    ]
    assert (output_dir / "combined.json").read_bytes().endswith(b"\n")


def test_combine_metadata_files_supports_legacy_chunked_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    for idx in range(1, 4):
        _write_json(input_dir / f"AF-{idx:04d}.json", {"model_id": f"AF-{idx:04d}"})

    outputs = combine_metadata_files(input_dir=input_dir, output_dir=output_dir, output_prefix="AF-chain", chunk_size=2)

    assert outputs == (
        (output_dir / "AF-chain-1-of-2.json").resolve(),
        (output_dir / "AF-chain-2-of-2.json").resolve(),
    )
    assert json.loads(outputs[0].read_text()) == [{"model_id": "AF-0001"}, {"model_id": "AF-0002"}]
    assert json.loads(outputs[1].read_text()) == [{"model_id": "AF-0003"}]


def test_combine_metadata_empty_input_fails_and_finalize_stops(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    with pytest.raises(MetadataCombineError):
        combine_metadata_files(input_dir=input_dir, output_dir=tmp_path / "outputs", output_filename="combined.json")

    work_dir = tmp_path / "work"
    (work_dir / "model_jsons").mkdir(parents=True)
    _write_json(work_dir / "chain_jsons" / "AF-0001.json", {"model_id": "AF-0001"})
    plan = plan_metadata_finalization(
        work_dir=work_dir,
        output_base_dir=tmp_path / "out",
        logical_shard_id=0,
        total_shards=1,
    )

    result = finalize_metadata(plan)

    assert result.exit_code == 1
    assert result.output_files == ()
    assert not (tmp_path / "out" / "metadata" / "collection" / "AF-chain-metadata-1-of-1.json").exists()


def test_finalize_metadata_partial_failure_reports_completed_outputs(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    _write_json(work_dir / "model_jsons" / "AF-0001.json", {"model_id": "AF-0001"})
    _write_json(work_dir / "chain_jsons" / "AF-0001.json", {"model_id": "AF-0001", "chain": "A"})
    plan = plan_metadata_finalization(
        work_dir=work_dir,
        output_base_dir=tmp_path / "out",
        logical_shard_id=0,
        total_shards=1,
    )

    def runner(combine_plan: MetadataCombinePlan) -> int:
        if "collection" in combine_plan.output_dir.parts:
            return 5
        combine_plan.output_dir.mkdir(parents=True, exist_ok=True)
        (combine_plan.output_dir / combine_plan.output_filename).write_text("[]\n")
        return 0

    result = finalize_metadata(plan, runner=runner)

    assert result.exit_code == 5
    assert result.output_files == (tmp_path / "out" / "metadata" / "search" / "AF-metadata-1-of-1.json",)
    assert result.output_files[0].exists()
    assert not (tmp_path / "out" / "metadata" / "collection" / "AF-chain-metadata-1-of-1.json").exists()


def test_real_toolkit_combine_metadata_contract_when_available(tmp_path: Path) -> None:
    if not REAL_TOOLKIT_COMBINE_METADATA.exists():
        reason = f"real combine_metadata.py not found at {REAL_TOOLKIT_COMBINE_METADATA}"
        if os.environ.get(REQUIRE_REAL_TOOLKIT_ENV_VAR) == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    input_dir = tmp_path / "inputs"
    toolkit_output_dir = tmp_path / "toolkit_outputs"
    native_output_dir = tmp_path / "native_outputs"
    output_prefix = "AF-chain-metadata-wpn7"
    chunk_size = 10_000
    expected_output = f"{output_prefix}-1-of-1.json"
    _write_json(
        input_dir / "AF-0002.json",
        [{"model_id": "AF-0002", "chain": "A"}, {"model_id": "AF-0002", "chain": "B"}],
    )
    _write_json(input_dir / "AF-0001.json", {"model_id": "AF-0001", "chain": "A"})

    result = subprocess.run(
        [
            sys.executable,
            str(REAL_TOOLKIT_COMBINE_METADATA),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(toolkit_output_dir),
            "--output-prefix",
            output_prefix,
            "--chunk-size",
            str(chunk_size),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    combine_metadata_files(
        input_dir=input_dir,
        output_dir=native_output_dir,
        output_prefix=output_prefix,
        chunk_size=chunk_size,
    )
    assert (native_output_dir / expected_output).read_bytes() == (toolkit_output_dir / expected_output).read_bytes()


def test_plan_metadata_finalization_skips_missing_inputs(tmp_path: Path) -> None:
    (tmp_path / "work" / "model_jsons").mkdir(parents=True)

    plan = plan_metadata_finalization(
        work_dir=tmp_path / "work",
        output_base_dir=tmp_path / "out",
        logical_shard_id=0,
        total_shards=1,
    )

    assert plan.search is not None
    assert plan.collection is None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
