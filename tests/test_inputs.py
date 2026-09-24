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

"""Tests for input preparation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from pytest import MonkeyPatch

from bspp.orchestration.contract.runspec import load_runspec
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.inputs import archives as archives_module
from bspp.orchestration.runtime.inputs.archives import (
    archives_for_dataset,
    archives_for_runspec_staging,
    archives_from_staging_dir,
    plan_archive_staging,
)
from bspp.orchestration.runtime.inputs.references import (
    ReferenceArtifact,
    check_references,
    ensure_references,
    plan_reference_downloads,
    references_from_spec,
)
from bspp.orchestration.runtime.inputs.reports import report_to_json, report_to_text


def test_reference_present_missing_hash_size(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_text("abc")
    missing = tmp_path / "missing.txt"
    digest = hashlib.sha256(b"abc").hexdigest()

    statuses = check_references(
        (
            ReferenceArtifact("present", present, expected_size=3, expected_sha256=digest),
            ReferenceArtifact("bad_size", present, expected_size=4),
            ReferenceArtifact("missing", missing),
        )
    )

    assert statuses[0].present is True
    assert statuses[0].ok is True
    assert statuses[0].size_ok is True
    assert statuses[0].sha256_ok is True
    assert statuses[1].ok is False
    assert statuses[1].size_ok is False
    assert statuses[2].present is False
    assert statuses[2].ok is False


def test_reference_dry_run_plans_no_writes_and_redacts_uri(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "ref.parquet"
    artifact = ReferenceArtifact(
        "tracking_parquet",
        destination,
        uri="https://user:secret@example.test/path/ref.parquet?token=secret",
    )

    plans = ensure_references((artifact,), dry_run=True)

    assert len(plans) == 1
    assert plans[0].scheme == "https"
    assert not destination.exists()
    assert not destination.parent.exists()
    assert "secret" not in json.dumps(plans[0].to_redacted_dict())


@dataclass(frozen=True)
class _FutureArtifact:
    local_path: Path
    uri: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _FutureReferences:
    master_parquet: _FutureArtifact


def test_references_from_spec_accepts_future_artifact_metadata(tmp_path: Path) -> None:
    path = tmp_path / "master.parquet"
    artifact = _FutureArtifact(path, "s3://bucket/master.parquet", 12, "abc")

    refs = references_from_spec(_FutureReferences(master_parquet=artifact))

    assert refs == (
        ReferenceArtifact(
            "master_parquet",
            path,
            uri="s3://bucket/master.parquet",
            expected_size=12,
            expected_sha256="abc",
        ),
    )


def test_references_from_runspec_use_artifact_metadata(tmp_path: Path) -> None:
    path = _write_input_runspec(tmp_path)
    spec = load_runspec(path)

    refs = references_from_spec(spec)

    manifest = next(ref for ref in refs if ref.name == "manifest_csv")
    assert manifest.uri == "gs://example/manifest.csv"
    assert manifest.expected_size == 1


def test_plan_reference_downloads_uses_path_fallback(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    artifact = ReferenceArtifact("manifest_csv", path)

    plans = plan_reference_downloads((artifact,))

    assert plans[0].source == str(path)
    assert plans[0].scheme == "local"
    assert plans[0].action == "missing"


def test_archives_for_dataset_counts_unique_and_missing_rows(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1", "ds1", "ds1", "ds2", "ds1"],
                "swiftstack_archive": ["b.tar.lz4", "a.tar.lz4", "a.tar.lz4", "z.tar.lz4", None],
            }
        ),
        tracking,
    )

    coverage = archives_for_dataset(tracking, "ds1")

    assert coverage.archives == ("a.tar.lz4", "b.tar.lz4")
    assert coverage.unique_archive_count == 2
    assert coverage.total_rows == 4
    assert coverage.missing_archive_rows == 1


def test_archives_for_dataset_prunes_columns_and_filters_scan(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1", "ds1", "ds2"],
                "swiftstack_archive": ["a.tar.lz4", "", "z.tar.lz4"],
                "model_entity_id": ["AF-0000000000000001", "AF-0000000000000002", "AF-0000000000000099"],
                "large_unused_payload": ["x" * 100, "y" * 100, "z" * 100],
            }
        ),
        tracking,
    )
    calls: list[dict[str, object]] = []
    original_read_table = archives_module.pq.read_table

    def spy_read_table(*args: object, **kwargs: object) -> pa.Table:
        calls.append(kwargs)
        return original_read_table(*args, **kwargs)

    monkeypatch.setattr(archives_module.pq, "read_table", spy_read_table)

    coverage = archives_for_dataset(tracking, "ds1")

    assert coverage.archives == ("a.tar.lz4",)
    assert coverage.total_rows == 2
    assert coverage.missing_archive_rows == 1
    assert calls == [
        {
            "columns": ["dataset_name", "swiftstack_archive"],
            "filters": [("dataset_name", "=", "ds1")],
        }
    ]


def test_archives_from_staging_dir_lists_existing_tar_archives(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "b.tar.lz4").write_text("b")
    (staging / "a.tar.lz4").write_text("a")
    (staging / "ignore.txt").write_text("ignored")

    coverage = archives_from_staging_dir(staging, "ds1")

    assert coverage.dataset == "ds1"
    assert coverage.archives == ("a.tar.lz4", "b.tar.lz4")
    assert coverage.total_rows == 2
    assert coverage.missing_archive_rows == 0


def test_archives_for_runspec_staging_selects_dataset_array_subset(tmp_path: Path) -> None:
    path = _write_input_runspec(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["dataset"]["array"] = "1-1"
    tracking = Path(data["references"]["tracking_parquet"])
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1", "ds1", "ds1"],
                "swiftstack_archive": ["a.tar.lz4", "b.tar.lz4", "c.tar.lz4"],
            }
        ),
        tracking,
    )
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    spec = load_runspec(path)

    full = archives_module.archives_for_runspec(spec)
    staged = archives_for_runspec_staging(spec, full)

    assert full.archives == ("a.tar.lz4", "b.tar.lz4", "c.tar.lz4")
    assert staged.archives == ("b.tar.lz4",)
    assert staged.total_rows == 1


def test_archives_for_runspec_staging_rejects_array_outside_tracking_coverage(tmp_path: Path) -> None:
    path = _write_input_runspec(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["dataset"]["array"] = "2-2"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    spec = load_runspec(path)

    with pytest.raises(ValueError, match="selects archive index 2"):
        archives_for_runspec_staging(spec)


def test_archive_staging_skip_and_download_plan(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "a.tar.lz4").write_text("staged")

    plan = plan_archive_staging(
        ["b.tar.lz4", "a.tar.lz4", "b.tar.lz4"],
        staging,
        archive_prefix="s3://example-bucket/structures",
        dry_run=True,
    )

    assert [item.archive for item in plan.items] == ["a.tar.lz4", "b.tar.lz4"]
    assert plan.items[0].action == "skip"
    assert plan.items[1].action == "download"
    assert plan.items[1].source == "s3://example-bucket/structures/b.tar.lz4"
    assert plan.present_count == 1
    assert plan.planned_count == 1


def test_archive_staging_exposes_non_mutating_s5cmd_command_file_plan(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    plan = plan_archive_staging(
        ["a.tar.lz4", "b.tar.lz4"],
        staging,
        archive_prefix="s3://token:secret@bspp/structures",
        s5cmd_path="/opt/bin/s5cmd-pdx",
        s5cmd_numworkers=24,
        command_file=tmp_path / "logs" / "downloads.txt",
    )

    command_file = plan.download_command_file

    assert command_file is not None
    assert command_file.s5cmd_path == "/opt/bin/s5cmd-pdx"
    assert command_file.argv == (
        "/opt/bin/s5cmd-pdx",
        "--numworkers",
        "24",
        "run",
        str(tmp_path / "logs" / "downloads.txt"),
    )
    assert command_file.commands == (
        f"cp s3://token:secret@bspp/structures/a.tar.lz4 {staging / 'a.tar.lz4'}",
        f"cp s3://token:secret@bspp/structures/b.tar.lz4 {staging / 'b.tar.lz4'}",
    )
    assert not command_file.command_file.exists()
    rendered = json.dumps(plan.to_redacted_dict())
    assert "s5cmd-pdx" in rendered
    assert "secret" not in rendered


def test_report_serialization_is_deterministic_and_json_serializable(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.parquet"
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1", "ds1"],
                "swiftstack_archive": ["b.tar.lz4", "a.tar.lz4"],
            }
        ),
        tracking,
    )
    coverage = archives_for_dataset(tracking, "ds1")
    plan = plan_archive_staging(coverage, tmp_path / "staging", archive_prefix="s3://token@bucket/prefix")
    report = {"staging": plan, "coverage": coverage}

    rendered_a = report_to_json(report)
    rendered_b = report_to_json(report)
    text = report_to_text(report)

    assert rendered_a == rendered_b
    decoded = json.loads(rendered_a)
    assert decoded["coverage"]["archives"] == ["a.tar.lz4", "b.tar.lz4"]
    assert "token" not in rendered_a
    assert "coverage:" in text


def test_inputs_prepare_cli_dry_run_reports_without_secret_values(tmp_path: Path) -> None:
    path = _write_input_runspec(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["inputs", "prepare", "--runspec", str(path)],
        env={
            "AWS_ACCESS_KEY_ID": "id-value",
            "AWS_SECRET_ACCESS_KEY": "secret-value",
            "S3_ENDPOINT_URL": "https://s3.example.test",
        },
    )

    assert result.exit_code == 0
    assert "archive_coverage:" in result.output
    assert "staging:" in result.output
    assert "AWS_SECRET_ACCESS_KEY" in result.output
    assert "secret-value" not in result.output


def _write_input_runspec(tmp_path: Path) -> Path:
    master = tmp_path / "master.parquet"
    tracking = tmp_path / "tracking.parquet"
    manifest = tmp_path / "manifest.csv"
    uniprot = tmp_path / "uniprot.duckdb"
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    (staging / "a.tar.lz4").write_text("archive")
    manifest.write_text("model_entity_id\nAF-1\n")
    uniprot.write_text("duckdb")
    pq.write_table(
        pa.table(
            {
                "source_run": ["ds1"],
                "swiftstack_archive": ["a.tar.lz4"],
                "model_entity_id": ["AF-1"],
            }
        ),
        master,
    )
    pq.write_table(
        pa.table(
            {
                "dataset_name": ["ds1"],
                "swiftstack_archive": ["a.tar.lz4"],
            }
        ),
        tracking,
    )
    data = {
        "dataset": {"name": "ds1", "run_id": "run", "mode": "archive", "array": "0-0"},
        "cluster": {"name": "example-cluster", "account": "acct"},
        "paths": {
            "project_root": str(tmp_path),
            "staging_dir": str(staging),
            "output_dir": str(output),
            "log_dir": str(output / "logs"),
            "legacy_repo": str(tmp_path / "legacy"),
            "orchestration_repo": str(tmp_path / "orch"),
        },
        "references": {
            "master_parquet": str(master),
            "tracking_parquet": str(tracking),
            "manifest_csv": {"path": str(manifest), "source_uri": "gs://example/manifest.csv", "min_size_bytes": 1},
            "uniprot_duckdb": str(uniprot),
        },
        "container": {"image": "image", "workdir": "/work", "mounts": []},
        "resources": {
            "gpu_worker": {
                "partition": "gpu",
                "cpus_per_task": 1,
                "memory": "1G",
                "time": "00:01:00",
                "array": "0-0",
            }
        },
        "worker": {
            "stages": "metadata_export",
            "workers": 1,
            "batch_size": 1,
            "shards_per_archive": 1,
            "self_upload": True,
            "local_scratch": True,
            "scratch_dir": "/dev/shm",
            "s5cmd_path": "/bin/s5cmd",
            "upload_slots": 1,
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/users/test/postprocessed/ds1/",
            "gcs_destination_prefix": None,
            "allow_production_prefixes": False,
        },
        "validation": {"expected_archives": 1},
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }
    path = tmp_path / "runspec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
