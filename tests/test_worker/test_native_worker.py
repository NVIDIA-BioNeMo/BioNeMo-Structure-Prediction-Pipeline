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

import csv
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bspp.orchestration.contract.runspec import RunSpec
from bspp.orchestration.runtime.worker import (
    MetadataCombinePlan,
    PipelineCommand,
    PipelineResult,
    TaskContext,
    WorkerDependencies,
    load_retry_failed_ids_from_state,
    native_worker,
    run_archive_task,
)

FIELDNAMES = ("model_entity_id", "entity_id", "chain_id", "uniprot_ac", "source")


def test_run_archive_task_composes_local_tar_batches_metadata_and_markers(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_ids = tuple(f"AF-{idx:016d}" for idx in range(1, 5))
    archive_name = "bspp_fixture_00001.tar.lz4"
    spec = _runspec(
        tmp_path,
        upload_mode="tar",
        local_tar_dir=tmp_path / "output" / "local_tars",
        local_tar_manifest_csv=tmp_path / "output" / "local_tars.csv",
        batch_size=1,
        analysis_metadata_csv=tmp_path / "output" / "analysis_metadata.csv",
    )
    _write_manifest(spec.references.manifest_csv, model_ids)
    (spec.paths.output_dir / "allowlists").mkdir(parents=True)
    (spec.paths.output_dir / "allowlists" / f"{archive_name}.txt").write_text("\n".join(model_ids) + "\n")
    calls: list[PipelineCommand] = []

    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models((*model_ids, "AF-9999999999999999")),
            pipeline_runner=_pipeline_runner(calls),
            metadata_runner=_metadata_runner,
            zstd_compressor=_fake_zstd,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert result.exit_code == 0
    assert result.archive_name == archive_name
    assert result.logical_shards == (0, 1)
    assert result.processed_models == 4
    assert result.failed_models == ()
    assert len(calls) == 4
    assert all("--no-cache" in call.argv for call in calls)

    tar_rows = _csv_rows(spec.storage.local_tar_manifest_csv)
    assert [row["tar_type"] for row in tar_rows] == ["batch", "batch", "metadata", "batch", "batch", "metadata"]
    assert [row["tar_name"] for row in tar_rows] == [
        "shard_0_batch_0.tar",
        "shard_0_batch_1.tar",
        "shard_0_metadata.tar",
        "shard_1_batch_0.tar",
        "shard_1_batch_1.tar",
        "shard_1_metadata.tar",
    ]
    assert {row["compression"] for row in tar_rows} == {"zstd-members"}
    assert all(row["s3_uri"].startswith("file://") for row in tar_rows)
    assert [row["task_id"] for row in tar_rows if row["shard_id"] == "1"] == ["0", "0", "1"]

    marker = json.loads((spec.paths.output_dir / "shard_0" / ".uploaded").read_text())
    assert marker["upload_mode"] == "tar"
    assert marker["total_batches"] == 2
    assert marker["total_files"] == 3
    assert marker["metadata_files"] == [str(spec.storage.local_tar_dir / "metadata" / "shard_0_metadata.tar")]
    assert not any((spec.paths.output_dir / "shard_0" / "success_outputs").rglob("*"))
    analysis_rows = _csv_rows(spec.analysis_metadata.csv_path)
    assert sorted(row["model_id"] for row in analysis_rows) == list(model_ids)
    assert {row["upload_status"] for row in analysis_rows} == {"local_tarred"}
    assert all(len(json.loads(row["expected_output_files_json"])) == 7 for row in analysis_rows)


def test_run_archive_task_reruns_partial_uploaded_local_tar_marker(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_id = "AF-0000000000000001"
    archive_name = "bspp_fixture_00001.tar.lz4"
    spec = _runspec(
        tmp_path,
        upload_mode="tar",
        local_tar_dir=tmp_path / "output" / "local_tars",
        local_tar_manifest_csv=tmp_path / "output" / "local_tars.csv",
        batch_size=1,
    )
    _write_manifest(spec.references.manifest_csv, (model_id,))
    shard_dir = spec.paths.output_dir / "shard_0"
    shard_dir.mkdir(parents=True)
    (shard_dir / ".uploaded").write_text(json.dumps({"status": "partial_uploaded", "model_count": 1}))
    calls: list[PipelineCommand] = []

    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models((model_id,)),
            pipeline_runner=_pipeline_runner(calls),
            metadata_runner=_metadata_runner,
            zstd_compressor=_fake_zstd,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    marker = json.loads((shard_dir / ".uploaded").read_text())
    assert marker["status"] == "uploaded"


def test_run_archive_task_refuses_local_tar_restart_with_stale_ledger_rows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_id = "AF-0000000000000001"
    archive_name = "bspp_fixture_00001.tar.lz4"
    spec = _runspec(
        tmp_path,
        upload_mode="tar",
        local_tar_dir=tmp_path / "output" / "local_tars",
        local_tar_manifest_csv=tmp_path / "output" / "local_tars.csv",
        batch_size=1,
    )
    _write_manifest(spec.references.manifest_csv, (model_id,))
    spec.storage.local_tar_manifest_csv.write_text(
        "timestamp,run_name,tar_type,s3_uri,tar_name,source_archive,shard_id,task_id,batch_id,member_count,size_bytes,compression\n"
        "t,run,batch,file://x,batch.tar,archive.tar.lz4,0,0,0,1,1,zstd-members\n"
    )

    with pytest.raises(RuntimeError, match="Refusing local-tar restart"):
        run_archive_task(
            spec,
            TaskContext(job_id="job1", array_task_id=0),
            WorkerDependencies(
                archive_names=(archive_name,),
                archive_extractor=_extractor_with_models((model_id,)),
                pipeline_runner=_pipeline_runner([]),
                metadata_runner=_metadata_runner,
                zstd_compressor=_fake_zstd,
            ),
        )


def test_run_archive_task_s3_tar_manifest_preserves_metadata_task_ids(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_ids = tuple(f"AF-{idx:016d}" for idx in range(1, 5))
    archive_name = "bspp_fixture_00001.tar.lz4"
    spec = _runspec(tmp_path, upload_mode="tar", batch_size=1, self_upload=True)
    _write_manifest(spec.references.manifest_csv, model_ids)

    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models(model_ids),
            pipeline_runner=_pipeline_runner([]),
            metadata_runner=_metadata_runner,
            upload_runner=lambda _argv: 0,
            zstd_compressor=_fake_zstd,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert result.exit_code == 0
    tar_rows = _csv_rows(spec.storage.s3_tar_manifest_csv)
    assert [row["tar_type"] for row in tar_rows] == ["batch", "batch", "metadata", "batch", "batch", "metadata"]
    assert [row["task_id"] for row in tar_rows if row["shard_id"] == "1"] == ["0", "0", "1"]


def test_run_archive_task_records_pipeline_failure_and_retry_marker_prefix(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_ids = ("AF-0000000000000001",)
    archive_name = "bspp_fixture_00001.tar.lz4"
    spec = _runspec(tmp_path, batch_size=2, retry_failed_only=True)
    _write_manifest(spec.references.manifest_csv, model_ids)
    shard_dir = spec.paths.output_dir / "shard_0"
    shard_dir.mkdir(parents=True)
    (shard_dir / "failed_models.tsv").write_text(f"{model_ids[0]}\tmodel\tprevious\n")

    calls: list[PipelineCommand] = []

    def failing_then_success(command: PipelineCommand) -> PipelineResult:
        calls.append(command)
        _write_pipeline_outputs(command, [model_ids[0]])
        return PipelineResult(exit_code=0, elapsed_seconds=0.1)

    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models(model_ids),
            pipeline_runner=failing_then_success,
            metadata_runner=_metadata_runner,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert result.exit_code == 0
    assert result.processed_models == 1
    assert len(calls) == 1
    assert (spec.paths.output_dir / "shard_0" / ".retry_failed_batch_0_done").exists()
    assert not (spec.paths.output_dir / "shard_0" / ".batch_0_done").exists()
    assert (
        spec.paths.output_dir
        / "shard_0"
        / "success_outputs"
        / "metadata"
        / "search"
        / "AF-metadata-1-of-2-20251128_full_run_v2-retry_failed_delta.json"
    ).exists()


def test_load_retry_failed_ids_prefers_latest_processing_log_over_failed_file(tmp_path: Path) -> None:
    processing_log = tmp_path / "processing_log.csv"
    with processing_log.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model_id", "original_id", "upload_status", "tar_of_origin", "shard_id", "task_id"),
        )
        writer.writeheader()
        writer.writerow({"model_id": "AF-1", "original_id": "AF-1", "upload_status": "model_failed"})
        writer.writerow({"model_id": "AF-1", "original_id": "AF-1", "upload_status": "uploaded"})
        writer.writerow({"model_id": "AF-2", "original_id": "AF-2", "upload_status": "pipeline_failed"})
    failed_file = tmp_path / "failed_models.tsv"
    failed_file.write_text("AF-3\tmodel\told\n")

    assert load_retry_failed_ids_from_state(processing_log, failed_file) == frozenset({"AF-2"})
    assert load_retry_failed_ids_from_state(tmp_path / "missing.csv", failed_file) == frozenset({"AF-3"})


def test_analysis_metadata_rows_collect_scores_and_quality_thresholds(tmp_path: Path) -> None:
    model_id = "AF-0000000000000001"
    spec = _runspec(tmp_path, analysis_metadata_csv=tmp_path / "analysis_metadata.csv")
    work_dir = tmp_path / "work"
    ipsae = work_dir / "ipsae" / "ipsae_summary.csv"
    ipsae.parent.mkdir(parents=True)
    ipsae.write_text(
        f"pdb_path,ipsae,pDockQ2,processing_time_ms,other\n{model_id}-model_v1.pdb,0.7,0.31,12,kept\n",
    )
    clashes = work_dir / "clash_interface_analysis" / f"{model_id}-model_v1_clashes.json"
    interfaces = work_dir / "clash_interface_analysis" / f"{model_id}-model_v1_interface.json"
    clashes.parent.mkdir(parents=True)
    clashes.write_text(
        json.dumps(
            {
                "sites": [
                    {"label": "backbone_clashes", "additional_site_annotations": {"n_clashes": 2}},
                    {"label": "heavy_atom_clashes", "additional_site_annotations": {"n_clashes": 5}},
                ],
            },
        ),
    )
    interfaces.write_text(
        json.dumps(
            {
                "sites": [
                    {"additional_site_annotations": {"interactions": ["a", "b"]}},
                    {"additional_site_annotations": {"interactions": ["c"]}},
                ],
            },
        ),
    )

    scores = native_worker._collect_analysis_scores(work_dir)
    rows = native_worker._analysis_metadata_rows(
        WorkerDependencies(now=lambda: datetime(2026, 5, 7, tzinfo=UTC)),
        spec,
        work_dir=work_dir,
        original_ids=(model_id,),
        model_ids=(model_id,),
        status_by_id={model_id: "uploaded"},
        failure_reasons={},
        archive_name="archive.tar.lz4",
        logical_shard_id=0,
        task_id=0,
        batch_id=0,
        batch_started_at="start",
        batch_finished_at="finish",
        include_scores=True,
        scores_by_id=scores,
    )

    assert rows[0]["passes_quality_threshold"] == "true"
    assert rows[0]["ipsae_max"] == 0.7
    assert rows[0]["pdockq2_max"] == 0.31
    score_payload = json.loads(rows[0]["scores_json"])
    assert score_payload["N_clash_backbone"] == 2
    assert score_payload["N_clash_heavyAtom"] == 5
    assert score_payload["N_interface_interactions"] == 3
    assert json.loads(rows[0]["expected_output_files_json"]) == [
        f"{model_id}-model_v1.cif",
        f"{model_id}-model_v1.pdb",
        f"{model_id}-model_v1.bcif",
        f"{model_id}-confidence_v1.json",
        f"{model_id}-predicted_aligned_error_v1.json",
        f"{model_id}-model_v1_clashes.json",
        f"{model_id}-model_v1_interface.json",
    ]


def test_run_archive_task_relinks_heterodimer_inputs_and_persists_rewritten_manifest(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    archive_name = "bspp_fixture_00001.tar.lz4"
    compound_id = "AF_1001_AF_1002"
    unified_id = "AF-7777"
    spec = _runspec(tmp_path, batch_size=1, heterodimers=True)
    _write_manifest(
        spec.references.manifest_csv,
        (),
        rows=[
            {"model_entity_id": "AF-1001", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P1", "source": "a"},
            {"model_entity_id": "AF-1002", "entity_id": "1", "chain_id": "A", "uniprot_ac": "P2", "source": "b"},
        ],
    )
    _write_manifest(
        spec.references.heterodimer_id_manifest,
        (),
        rows=[
            {"model_entity_id": unified_id, "entity_id": "1", "chain_id": "A", "uniprot_ac": "P1", "source": ""},
            {"model_entity_id": unified_id, "entity_id": "2", "chain_id": "B", "uniprot_ac": "P2", "source": ""},
        ],
    )
    saw_symlink = {"value": False}
    saw_model_ids: list[str] = []
    saw_manifest_ids: list[str] = []

    def runner(command: PipelineCommand) -> PipelineResult:
        input_dir = Path(command.argv[command.argv.index("--input-dir") + 1])
        output_dir = Path(command.argv[command.argv.index("--output-dir") + 1])
        chain_mapping = Path(command.argv[command.argv.index("--chain-mapping") + 1])
        saw_symlink["value"] = (input_dir / f"{unified_id}-meta_v1.json").is_symlink()
        saw_model_ids.extend((output_dir / "model_ids.txt").read_text().splitlines())
        saw_manifest_ids.extend(row["model_entity_id"] for row in _csv_rows(chain_mapping))
        _write_pipeline_outputs(command, saw_model_ids)
        return PipelineResult(exit_code=0, elapsed_seconds=0.1)

    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models((compound_id,)),
            pipeline_runner=runner,
            metadata_runner=_metadata_runner,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert result.exit_code == 0
    assert saw_symlink["value"] is True
    assert saw_model_ids == [unified_id]
    assert saw_manifest_ids == [unified_id, unified_id]
    table = pq.read_table(spec.paths.output_dir / "shard_0" / "shard_manifest.parquet")
    assert table.column("model_entity_id").to_pylist() == [unified_id, unified_id]
    processing_rows = _csv_rows(spec.paths.output_dir / "processing_log.csv")
    assert processing_rows[0]["original_id"] == compound_id
    assert processing_rows[0]["model_id"] == unified_id


def test_run_archive_task_uses_existing_container_mount_for_pipeline_script(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_id = "AF-0000000000000001"
    archive_name = "bspp_fixture_00001.tar.lz4"
    mounted_toolkit = tmp_path / "workspace" / "AFDB-Integration-Kit"
    pipeline_script = mounted_toolkit / "scripts" / "production_pipeline.py"
    pipeline_script.parent.mkdir(parents=True)
    pipeline_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    spec = _runspec(tmp_path, batch_size=1, toolkit_mount_target=mounted_toolkit)
    _write_manifest(spec.references.manifest_csv, (model_id,))
    calls: list[PipelineCommand] = []

    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models((model_id,)),
            pipeline_runner=_pipeline_runner(calls),
            metadata_runner=_metadata_runner,
            zstd_compressor=_fake_zstd,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert result.exit_code == 0
    assert calls[0].argv[0] == sys.executable
    assert calls[0].argv[calls[0].argv.index("--python-cmd") + 1] == sys.executable
    assert calls[0].argv[1] == str(pipeline_script)
    assert ("DUCKDB_MEMORY_LIMIT", "1GB") in calls[0].env


@pytest.mark.parametrize(
    ("python_cmd", "expected_python_cmd"),
    (
        (None, sys.executable),
        ("", sys.executable),
        ("/opt/custom/bin/python", "/opt/custom/bin/python"),
    ),
)
def test_pipeline_options_use_active_interpreter_unless_explicitly_injected(
    tmp_path: Path,
    python_cmd: str | None,
    expected_python_cmd: str,
) -> None:
    options = native_worker._pipeline_options(
        _runspec(tmp_path),
        WorkerDependencies(python_cmd=python_cmd),
    )

    assert options.python_cmd == expected_python_cmd


def test_pipeline_subprocess_writes_captured_output_paths(tmp_path: Path) -> None:
    script = tmp_path / "pipeline.py"
    script.write_text(
        "import sys\nprint('pipeline stdout')\nprint('pipeline stderr', file=sys.stderr)\nsys.exit(2)\n",
        encoding="utf-8",
    )
    stdout_path = tmp_path / "logs" / "pipeline.stdout.txt"
    stderr_path = tmp_path / "logs" / "pipeline.stderr.txt"

    result = native_worker._run_pipeline_subprocess(
        PipelineCommand(
            argv=(sys.executable, str(script)),
            env=(("BSPP_PIPELINE_STDOUT", str(stdout_path)), ("BSPP_PIPELINE_STDERR", str(stderr_path))),
        )
    )

    assert result.exit_code == 2
    assert result.stdout_path == stdout_path
    assert result.stderr_path == stderr_path
    assert stdout_path.read_text(encoding="utf-8") == "pipeline stdout\n"
    assert stderr_path.read_text(encoding="utf-8") == "pipeline stderr\n"


def test_run_archive_task_self_upload_files_mode_success_and_fallback(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_id = "AF-0000000000000001"
    archive_name = "bspp_fixture_00001.tar.lz4"

    success_spec = _runspec(tmp_path / "success", batch_size=1, self_upload=True)
    _write_manifest(success_spec.references.manifest_csv, (model_id,))
    success_result = run_archive_task(
        success_spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models((model_id,)),
            pipeline_runner=_pipeline_runner([]),
            metadata_runner=_metadata_runner,
            upload_runner=lambda _argv: 0,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert success_result.exit_code == 0
    success_marker = json.loads((success_spec.paths.output_dir / "shard_0" / ".uploaded").read_text())
    assert success_marker["s3_prefix"] == "s3://example-bucket/test-output/"
    assert success_marker["metadata_files"] == [
        "AF-metadata-1-of-2-20251128_full_run_v2.json",
        "AF-chain-metadata-1-of-2-20251128_full_run_v2.json",
    ]
    assert not (success_spec.paths.output_dir / "shard_0" / "success_outputs").exists()

    fallback_spec = _runspec(tmp_path / "fallback", batch_size=1, self_upload=True)
    _write_manifest(fallback_spec.references.manifest_csv, (model_id,))
    calls = {"count": 0}

    def fail_batch_then_succeed_metadata(_argv: Sequence[str]) -> int:
        calls["count"] += 1
        return 1 if calls["count"] <= 3 else 0

    fallback_result = run_archive_task(
        fallback_spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models((model_id,)),
            pipeline_runner=_pipeline_runner([]),
            metadata_runner=_metadata_runner,
            upload_runner=fail_batch_then_succeed_metadata,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert fallback_result.exit_code == 0
    assert (fallback_spec.paths.output_dir / "shard_0" / "success_outputs" / f"{model_id}-model_v1.cif").exists()
    assert _csv_rows(fallback_spec.paths.output_dir / "processing_log.csv")[0]["upload_status"] == (
        "upload_failed_lustre_fallback"
    )


def test_run_archive_task_self_upload_metadata_upload_failure_still_writes_marker(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_id = "AF-0000000000000001"
    archive_name = "bspp_fixture_00001.tar.lz4"
    spec = _runspec(tmp_path, batch_size=1, self_upload=True)
    _write_manifest(spec.references.manifest_csv, (model_id,))
    calls = {"count": 0}

    def fail_metadata_upload(_argv: Sequence[str]) -> int:
        calls["count"] += 1
        return 0 if calls["count"] == 1 else 1

    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models((model_id,)),
            pipeline_runner=_pipeline_runner([]),
            metadata_runner=_metadata_runner,
            upload_runner=fail_metadata_upload,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert result.exit_code == 0
    marker = json.loads((spec.paths.output_dir / "shard_0" / ".uploaded").read_text())
    assert marker["status"] == "uploaded"
    assert marker["metadata_files"] == []
    assert marker["total_files"] == 7


def test_run_archive_task_metadata_failure_skips_uploaded_marker(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_id = "AF-0000000000000001"
    archive_name = "bspp_fixture_00001.tar.lz4"
    spec = _runspec(tmp_path, batch_size=1, self_upload=True)
    _write_manifest(spec.references.manifest_csv, (model_id,))

    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=_extractor_with_models((model_id,)),
            pipeline_runner=_pipeline_runner([]),
            metadata_runner=lambda _command: 7,
            upload_runner=lambda _argv: 0,
            now=lambda: datetime(2026, 5, 7, tzinfo=UTC),
        ),
    )

    assert result.exit_code == 7
    assert not (spec.paths.output_dir / "shard_0" / ".uploaded").exists()


def test_run_archive_task_writes_and_removes_active_marker(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow.parquet")
    model_id = "AF-0000000000000001"
    archive_name = "bspp_fixture_00001.tar.lz4"
    spec = _runspec(
        tmp_path,
        upload_mode="tar",
        local_tar_dir=tmp_path / "output" / "local_tars",
        local_tar_manifest_csv=tmp_path / "output" / "local_tars.csv",
        batch_size=1,
    )
    _write_manifest(spec.references.manifest_csv, (model_id,))
    active_marker = spec.paths.output_dir / ".archive_active_0"
    marker_seen_during_extract: list[bool] = []

    def extractor(_archive_path: Path, destination: Path) -> None:
        marker_seen_during_extract.append(active_marker.exists())
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{model_id}-meta_v1.json").write_text('{"pae":[1],"plddt":[2],"max_pae":3}')
        (destination / f"{model_id}-model_v1.pdb").write_text("MODEL\n")

    calls: list[PipelineCommand] = []
    result = run_archive_task(
        spec,
        TaskContext(job_id="job1", array_task_id=0),
        WorkerDependencies(
            archive_names=(archive_name,),
            archive_extractor=extractor,
            pipeline_runner=_pipeline_runner(calls),
            metadata_runner=_metadata_runner,
            zstd_compressor=_fake_zstd,
        ),
    )

    assert result.exit_code == 0
    assert marker_seen_during_extract == [True]
    assert not active_marker.exists()


def _runspec(
    tmp_path: Path,
    *,
    upload_mode: str = "files",
    local_tar_dir: Path | None = None,
    local_tar_manifest_csv: Path | None = None,
    batch_size: int = 1,
    retry_failed_only: bool = False,
    heterodimers: bool = False,
    self_upload: bool = False,
    analysis_metadata_csv: Path | None = None,
    dataset_name: str = "20251128_full_run_v2",
    shards_per_archive: int = 2,
    toolkit_mount_target: Path | None = None,
) -> RunSpec:
    output_dir = tmp_path / "output"
    references = tmp_path / "refs"
    reference_data: dict[str, Any] = {
        "master_parquet": str(references / "master.parquet"),
        "tracking_parquet": str(references / "tracking.parquet"),
        "manifest_csv": str(references / "manifest.csv"),
        "uniprot_duckdb": str(references / "uniprot.duckdb"),
    }
    if heterodimers:
        reference_data["heterodimer_id_manifest"] = str(references / "heterodimer_ids.csv")

    data: dict[str, Any] = {
        "dataset": {"name": dataset_name, "run_id": "test-run", "mode": "archive", "array": "0-0"},
        "cluster": {"name": "example-cluster", "account": "example-account"},
        "paths": {
            "project_root": str(tmp_path),
            "staging_dir": str(tmp_path / "staging"),
            "output_dir": str(output_dir),
            "log_dir": str(tmp_path / "logs"),
            "legacy_repo": str(tmp_path / "AFDB-Integration-Kit"),
            "afdb_toolkit_repo": str(tmp_path / "AFDB-Integration-Kit"),
            "orchestration_repo": str(tmp_path / "bspp-orchestration"),
            "recipe_dir": str(tmp_path / "recipe"),
        },
        "references": reference_data,
        "container": {
            "image": "image.sqsh",
            "workdir": str(tmp_path),
            "mounts": [
                {
                    "source": str(tmp_path / "AFDB-Integration-Kit"),
                    "target": str(tmp_path / "mounted-toolkit"),
                    "read_only": True,
                },
            ],
        },
        "resources": {
            "gpu_worker": {
                "partition": "batch_singlenode",
                "cpus_per_task": 30,
                "memory": "128G",
                "time": "04:00:00",
                "gres": "gpu:1",
                "array": "0-0",
            }
        },
        "worker": {
            "stages": "metadata_export modelcif_export",
            "workers": 2,
            "batch_size": batch_size,
            "shards_per_archive": shards_per_archive,
            "self_upload": self_upload,
            "local_scratch": True,
            "scratch_dir": str(tmp_path / "scratch"),
            "s5cmd_path": "s5cmd",
            "upload_slots": 0,
            "retry_failed_only": retry_failed_only,
            "heterodimers": heterodimers,
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/test-output/",
            "upload_mode": upload_mode,
            "local_tar_dir": str(local_tar_dir) if local_tar_dir else None,
            "local_tar_manifest_csv": str(local_tar_manifest_csv) if local_tar_manifest_csv else None,
            "tar_compression": "zstd-members" if upload_mode == "tar" else "none",
            "allow_production_prefixes": False,
        },
        "analysis_metadata": {
            "enabled": analysis_metadata_csv is not None,
            "csv_path": str(analysis_metadata_csv) if analysis_metadata_csv else None,
            "finalize_after_gpu": False,
        },
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }
    if upload_mode == "tar" and self_upload:
        storage = data["storage"]
        assert isinstance(storage, dict)
        storage.update(
            {
                "s3_tar_prefix": "s3://example-bucket/test-output/tars/",
                "s3_tar_manifest_csv": str(tmp_path / "output" / "uploaded_tars.csv"),
            }
        )
    if toolkit_mount_target is not None:
        container = data["container"]
        assert isinstance(container, dict)
        mounts = container.get("mounts", [])
        assert isinstance(mounts, list)
        mounts[:] = [{"source": str(tmp_path / "AFDB-Integration-Kit"), "target": str(toolkit_mount_target)}]
        container["mounts"] = mounts
    # Create the override toolkit tree at the container target path so
    # resolve_toolkit (filesystem-checking variant) can validate it.
    _ensure_override_toolkit_tree(tmp_path, toolkit_mount_target)
    spec = RunSpec.from_mapping(data)
    spec.paths.staging_dir.mkdir(parents=True, exist_ok=True)
    (spec.paths.staging_dir / "bspp_fixture_00001.tar.lz4").write_text("archive")
    spec.paths.output_dir.mkdir(parents=True, exist_ok=True)
    references.mkdir(parents=True)
    spec.references.uniprot_duckdb.write_text("duckdb")
    return spec


def _ensure_override_toolkit_tree(tmp_path: Path, toolkit_mount_target: Path | None = None) -> None:
    """Create a minimal override toolkit tree at the container target path.

    The native worker uses ``resolve_toolkit`` (filesystem-checking variant)
    which requires the container target path to exist with required files.
    """
    target = toolkit_mount_target or (tmp_path / "mounted-toolkit")
    (target / "scripts").mkdir(parents=True, exist_ok=True)
    (target / "scripts" / "production_pipeline.py").write_text("#!/usr/bin/env python3\n")
    (target / "pyproject.toml").write_text("[project]\nname='afdb-toolkit'\n")


def _extractor_with_models(model_ids: Sequence[str]):
    def extract(_archive_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for model_id in model_ids:
            (destination / f"{model_id}-meta_v1.json").write_text('{"pae":[1],"plddt":[2],"max_pae":3}')
            (destination / f"{model_id}-model_v1.pdb").write_text("MODEL\n")

    return extract


def _pipeline_runner(calls: list[PipelineCommand]):
    def run(command: PipelineCommand) -> PipelineResult:
        calls.append(command)
        ids = (Path(command.argv[command.argv.index("--output-dir") + 1]) / "model_ids.txt").read_text().splitlines()
        _write_pipeline_outputs(command, ids)
        return PipelineResult(exit_code=0, elapsed_seconds=0.1)

    return run


def _write_pipeline_outputs(command: PipelineCommand, model_ids: Sequence[str]) -> None:
    output_dir = Path(command.argv[command.argv.index("--output-dir") + 1])
    for model_id in model_ids:
        for dirname, suffix in (
            ("modelcif", "-model_v1.cif"),
            ("modelpdb", "-model_v1.pdb"),
            ("bcif", "-model_v1.bcif"),
            ("scores", "-confidence_v1.json"),
            ("scores", "-predicted_aligned_error_v1.json"),
            ("clash_interface_analysis", "-model_v1_clashes.json"),
            ("clash_interface_analysis", "-model_v1_interface.json"),
        ):
            path = output_dir / dirname / f"{model_id}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{model_id}{suffix}")
    for dirname in ("model_jsons", "chain_jsons"):
        for model_id in model_ids:
            path = output_dir / dirname / f"{model_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")


def _metadata_runner(plan: MetadataCombinePlan) -> int:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / plan.output_filename).write_text(plan.output_filename)
    return 0


def _fake_zstd(source: Path, destination: Path) -> None:
    destination.write_bytes(b"zstd:" + source.read_bytes())


def _write_manifest(
    path: Path | None,
    model_ids: Sequence[str],
    *,
    rows: Sequence[dict[str, str]] | None = None,
) -> None:
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = rows or [
        {
            "model_entity_id": model_id,
            "entity_id": "1",
            "chain_id": "A",
            "uniprot_ac": f"P{idx:05d}",
            "source": "fixture",
        }
        for idx, model_id in enumerate(model_ids, start=1)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest_rows)


def _csv_rows(path: Path | None) -> list[dict[str, str]]:
    assert path is not None
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- pdb-assembly worker-level discovery test (additive) ---


def test_discover_workspace_model_ids_finds_pdb_assembly_corpus_bundle(tmp_path: Path) -> None:
    """Pin that a corpus bundle with pdb-assembly members is discovered end-to-end
    through the _discover_workspace_model_ids layer (rglob -> relative_to -> sorted ->
    discover_model_ids). Before the pdb-assembly extension, this layer returned
    () for corpus bundles, causing run_archive_task to exit 1 (zero model IDs).
    After the extension, the pdb-assembly IDs are discovered and the worker proceeds.
    The test pins the _discover_workspace_model_ids output; it does not drive
    run_archive_task or assert an exit code directly."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Flat-form corpus members (as extracted from a .tar.lz4 archive)
    (input_dir / "pdb_5snm_assembly_1-model_v1.pdb").write_text("dummy")
    (input_dir / "pdb_5snm_assembly_1-meta_v1.json").write_text("{}")
    (input_dir / "pdb_6snm_assembly_2-model_v1.pdb").write_text("dummy")

    model_ids = native_worker._discover_workspace_model_ids(input_dir)
    assert model_ids == ("pdb_5snm_assembly_1", "pdb_6snm_assembly_2")
