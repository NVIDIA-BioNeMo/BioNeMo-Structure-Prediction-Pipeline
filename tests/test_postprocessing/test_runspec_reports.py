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

"""Tests for RunSpec-driven archive output validation and cleanup reports."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.runspec import load_runspec
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.postprocessing.gcs_preflight import build_gcs_preflight_report
from bspp.orchestration.runtime.postprocessing.runspec_reports import (
    build_archive_output_validation_report,
    plan_cleanup,
    plan_fallback_upload,
)


def test_archive_output_validation_report_matches_one_archive_acceptance_counts(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    aggregate = tmp_path / "aggregate.parquet"
    pq.write_table(pa.table({"model_id": list(range(6730))}), aggregate)  # type: ignore[no-untyped-call]
    _make_uploaded_shard(spec.paths.output_dir, 0, model_count=2000, file_count=14000, manifest_rows=4000)
    _make_uploaded_shard(spec.paths.output_dir, 1, model_count=1365, file_count=9505, manifest_rows=2730)

    report = build_archive_output_validation_report(spec, aggregate_manifest=aggregate)

    assert report.valid is True
    assert report.expected_logical_shards_for_array == 2
    assert report.uploaded_marker_count == 2
    assert report.processed_models == 3365
    assert report.uploaded_objects == 23505
    assert report.aggregate_rows == 6730
    assert {check.name: check.ok for check in report.checks} == {
        "logical_shards_for_array": True,
        "uploaded_markers": True,
        "one_archive_processed_models": True,
        "one_archive_uploaded_objects": True,
        "one_archive_aggregate_rows": True,
    }


def test_archive_output_validation_report_counts_throttled_array(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, array="0-37%25"))

    report = build_archive_output_validation_report(spec)

    assert report.expected_archives_for_array == 38
    assert report.expected_logical_shards_for_array == 76


def test_fallback_upload_plan_is_marker_aware_and_runspec_scoped(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _make_uploaded_shard(spec.paths.output_dir, 0, model_count=1, file_count=2, manifest_rows=1, with_success=True)
    _make_success_outputs(spec.paths.output_dir, 1)

    plan = plan_fallback_upload(spec, tool="s5cmd")

    assert plan.destination_prefix == spec.storage.s3_output_prefix
    assert plan.shards_skipped_self_uploaded == 1
    assert plan.shards_queued == 1
    assert [path.parent.name for path in plan.queued_success_output_dirs] == ["shard_1"]
    assert plan.no_op is False


def test_fallback_upload_omitted_tool_defaults_to_s5cmd(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _make_success_outputs(spec.paths.output_dir, 0)

    plan = plan_fallback_upload(spec)

    assert plan.tool == "s5cmd"
    assert plan.shards_queued == 1


def test_gcs_preflight_omitted_tool_defaults_to_gcloud(tmp_path: Path) -> None:
    spec = load_runspec(
        _write_runspec(
            tmp_path,
            gcs_destination_prefix="gs://example-gcs-bucket/delivery/ds1/",
            gcs_credentials_ref="env:bspp/gcs",
        )
    )

    report = build_gcs_preflight_report(spec)

    assert report.tool == "gcloud"
    assert report.destination_prefix == "gs://example-gcs-bucket/delivery/ds1/"


def test_cleanup_plan_removes_marker_protected_success_outputs(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _make_uploaded_shard(spec.paths.output_dir, 0, model_count=1, file_count=2, manifest_rows=1, with_success=True)
    _make_uploaded_shard(spec.paths.output_dir, 1, model_count=1, file_count=2, manifest_rows=1, with_success=True)

    dry_run = plan_cleanup(spec)
    executed = plan_cleanup(spec, execute=True)

    assert dry_run.safe_to_execute is True
    assert dry_run.success_output_dirs_before == 2
    assert executed.cleaned_dirs == 2
    assert executed.success_output_dirs_after == 0
    assert not (spec.paths.output_dir / "shard_0" / "success_outputs").exists()


def test_runspec_validate_archive_output_cli_writes_neutral_report(tmp_path: Path) -> None:
    spec_path = _write_runspec(tmp_path)
    spec = load_runspec(spec_path)
    aggregate = tmp_path / "aggregate.parquet"
    pq.write_table(pa.table({"model_id": list(range(6730))}), aggregate)  # type: ignore[no-untyped-call]
    _make_uploaded_shard(spec.paths.output_dir, 0, model_count=2000, file_count=14000, manifest_rows=4000)
    _make_uploaded_shard(spec.paths.output_dir, 1, model_count=1365, file_count=9505, manifest_rows=2730)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "runspec",
            "validate-archive-output",
            str(spec_path),
            "--aggregate-manifest",
            str(aggregate),
            "--write-report",
            "--strict",
        ],
    )

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["valid"] is True
    saved = json.loads((spec.paths.output_dir / "wp6" / "archive_output_validation_report.json").read_text())
    assert saved["uploaded_objects"] == 23505
    assert (spec.paths.output_dir / "wp6" / "archive_output_validation_report.txt").is_file()


def test_runspec_validate_bridge_cli_is_removed(tmp_path: Path) -> None:
    spec_path = _write_runspec(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["runspec", "validate-bridge", str(spec_path)])

    assert result.exit_code != 0
    assert "No such command 'validate-bridge'" in result.output


def _make_uploaded_shard(
    output_dir: Path,
    shard_id: int,
    *,
    model_count: int,
    file_count: int,
    manifest_rows: int,
    with_success: bool = False,
) -> None:
    shard = output_dir / f"shard_{shard_id}"
    shard.mkdir(parents=True, exist_ok=True)
    (shard / ".uploaded").write_text(json.dumps({"model_count": model_count, "total_files": file_count}))
    (shard / ".batch_0_done").touch()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"model_id": list(range(manifest_rows))}),
        shard / "shard_manifest.parquet",
    )
    if with_success:
        _make_success_outputs(output_dir, shard_id)


def _make_success_outputs(output_dir: Path, shard_id: int) -> None:
    success = output_dir / f"shard_{shard_id}" / "success_outputs"
    success.mkdir(parents=True, exist_ok=True)
    (success / "model.pdb").write_text("MODEL\n")


def _write_runspec(
    tmp_path: Path,
    *,
    array: str = "0-0",
    gcs_destination_prefix: str | None = None,
    gcs_credentials_ref: str | None = None,
) -> Path:
    data = {
        "dataset": {"name": "ds1", "run_id": "run", "mode": "archive", "array": array},
        "cluster": {"name": "example-cluster", "account": "acct", "owner": "tester"},
        "paths": {
            "project_root": str(tmp_path),
            "staging_dir": str(tmp_path / "input" / "staging"),
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "output" / "logs"),
            "legacy_repo": str(tmp_path / "AFDB-Integration-Kit"),
            "orchestration_repo": str(tmp_path / "bspp-orchestration"),
        },
        "references": {
            "master_parquet": str(tmp_path / "master.parquet"),
            "tracking_parquet": str(tmp_path / "tracking.parquet"),
            "manifest_csv": str(tmp_path / "manifest.csv"),
            "uniprot_duckdb": str(tmp_path / "uniprot.duckdb"),
        },
        "container": {"image": "image.sqsh", "workdir": "/workspace/bspp-orchestration", "mounts": []},
        "resources": {
            "gpu_worker": {
                "partition": "gpu",
                "cpus_per_task": 30,
                "memory": "128G",
                "time": "04:00:00",
                "gres": "gpu:1",
                "array": "0-0",
            }
        },
        "worker": {
            "stages": "metadata_export",
            "workers": 24,
            "batch_size": 500,
            "shards_per_archive": 2,
            "self_upload": True,
            "local_scratch": True,
            "scratch_dir": "/dev/shm",
            "s5cmd_path": "/bin/s5cmd",
            "upload_slots": 4,
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/users/test/postprocessed/ds1/",
            "gcs_destination_prefix": gcs_destination_prefix,
            "allow_production_prefixes": False,
        },
        "validation": {
            "expected_archives": 38,
            "expected_one_archive_models": 3365,
            "expected_one_archive_objects": 23505,
            "expected_one_archive_aggregate_rows": 6730,
        },
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": gcs_credentials_ref},
    }
    path = tmp_path / "runspec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
