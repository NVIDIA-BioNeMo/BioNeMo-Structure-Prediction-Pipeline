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

"""Tests for archive cleanup planning and execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from bspp.orchestration.contract.runspec import load_runspec
from bspp.orchestration.runtime.cli import cli
from bspp.orchestration.runtime.postprocessing.archive_cleanup import (
    ArchiveCleanupReport,
    load_archive_names,
    plan_archive_cleanup,
    render_archive_cleanup_tsv,
)


def test_success_path_deletes_archive_on_execute(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=2, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)

    dry_run = plan_archive_cleanup(spec)
    assert [item.status for item in dry_run.items] == ["delete_ready"]
    assert dry_run.delete_ready_count == 1
    assert archive.exists()

    executed = plan_archive_cleanup(spec, execute=True)
    assert [item.status for item in executed.items] == ["deleted"]
    assert executed.deleted_count == 1
    assert not archive.exists()


def test_dry_run_does_not_delete(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)

    plan_archive_cleanup(spec, execute=False)

    assert archive.exists()


def test_idempotent_rerun_after_deletion(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)

    plan_archive_cleanup(spec, execute=True)
    rerun = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in rerun.items] == ["missing_archive"]
    assert rerun.deleted_count == 0


def test_missing_uploaded_marker_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    (spec.paths.output_dir / "shard_1").mkdir(parents=True, exist_ok=True)

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_missing_uploaded_marker"]
    assert archive.exists()


def test_partial_upload_status_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(
        spec.paths.output_dir,
        1,
        model_count=1,
        s3_prefix=spec.storage.s3_output_prefix,
        status="partial_uploaded",
    )

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_failed_models"]
    assert archive.exists()


def test_fallback_batch_missing_done_marker_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    # shard_1 claims 2 batches but only one .batch_*_done marker exists.
    _make_clean_shard(
        spec.paths.output_dir,
        1,
        model_count=1,
        s3_prefix=spec.storage.s3_output_prefix,
        total_batches=2,
        done_markers=1,
    )

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_missing_batch_marker"]
    assert archive.exists()


def test_partial_batch_marker_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    (spec.paths.output_dir / "shard_1" / ".batch_1_partial_uploaded").touch()

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_missing_batch_marker"]
    assert archive.exists()


def test_empty_metadata_files_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(
        spec.paths.output_dir,
        1,
        model_count=1,
        s3_prefix=spec.storage.s3_output_prefix,
        metadata_files=(),
    )

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_missing_metadata_upload"]
    assert archive.exists()


def test_prefix_mismatch_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix="s3://example-bucket/production/wrong/")

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_prefix_mismatch"]
    assert archive.exists()


def test_allowlist_model_count_mismatch_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=2, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _write_allowlist(spec.paths.output_dir, "a.tar.lz4", 4)

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_model_count_mismatch"]
    assert archive.exists()


def test_allowlist_model_count_match_deletes(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=2, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _write_allowlist(spec.paths.output_dir, "a.tar.lz4", 3)

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["deleted"]
    assert not archive.exists()


def test_missing_shard_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_missing_shard"]
    assert archive.exists()


def test_success_outputs_residue_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    success = spec.paths.output_dir / "shard_1" / "success_outputs"
    success.mkdir(parents=True, exist_ok=True)
    (success / "model.pdb").write_text("MODEL\n")

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_fallback_upload"]
    assert archive.exists()


def test_traversal_archive_entry_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["../escape.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    outside = tmp_path / "escape.tar.lz4"
    outside.write_text("OUTSIDE\n")

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_unsafe_archive_path"]
    assert outside.exists()


def test_symlinked_archive_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    target = tmp_path / "real.tar.lz4"
    target.write_text("REAL\n")
    spec.paths.staging_dir.mkdir(parents=True, exist_ok=True)
    (spec.paths.staging_dir / "a.tar.lz4").symlink_to(target)

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_unsafe_archive_path"]
    assert target.exists()


def test_active_worker_marker_refuses(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    (spec.paths.output_dir / ".archive_active_0").touch()

    report = plan_archive_cleanup(spec, execute=True)

    assert [item.status for item in report.items] == ["keep_worker_active"]
    assert archive.exists()


def test_shards_per_archive_mismatch_raises(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=3)

    with pytest.raises(ValueError, match="shards_per_archive"):
        plan_archive_cleanup(spec)


def test_non_self_upload_raises(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, self_upload=False))

    with pytest.raises(ValueError, match="self_upload"):
        plan_archive_cleanup(spec)


def test_staging_dir_archive_source_raises(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path, archive_source="staging_dir"))

    with pytest.raises(ValueError, match="archive_source 'tracking'"):
        plan_archive_cleanup(spec)


def test_empty_expected_prefix_raises(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)

    with pytest.raises(ValueError, match="must be non-empty"):
        plan_archive_cleanup(spec, expected_s3_prefix="  ")


def test_cli_plan_cleanup_archives_writes_json_and_tsv(tmp_path: Path) -> None:
    spec_path = _write_runspec(tmp_path)
    spec = load_runspec(spec_path)
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["runspec", "plan-cleanup-archives", str(spec_path), "--write-report"],
    )

    assert result.exit_code == 0
    rendered = json.loads(result.output)
    assert rendered["delete_ready_count"] == 1
    assert (spec.paths.output_dir / "archive_cleanup_report.json").is_file()
    tsv = (spec.paths.output_dir / "archive_cleanup.tsv").read_text()
    assert tsv.splitlines()[0].startswith("archive_name\tarchive_task_id\t")


def test_render_tsv_contains_expected_columns(tmp_path: Path) -> None:
    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)

    report = plan_archive_cleanup(spec)
    tsv = render_archive_cleanup_tsv(report)

    header = tsv.splitlines()[0].split("\t")
    assert header == [
        "archive_name",
        "archive_task_id",
        "logical_shards",
        "archive_path",
        "archive_size",
        "expected_model_count",
        "uploaded_model_count",
        "expected_s3_prefix",
        "uploaded_s3_prefixes",
        "status",
        "action",
        "reason",
    ]
    assert "delete_ready" in tsv


def test_load_archive_names_requires_string_list(tmp_path: Path) -> None:
    path = tmp_path / "archive_list.json"
    path.write_text(json.dumps([1, 2, 3]))

    with pytest.raises(ValueError, match="JSON list of strings"):
        load_archive_names(tmp_path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_cleanup_blocks_while_worker_lock_held(tmp_path: Path) -> None:
    import threading
    import time

    from bspp.orchestration.runtime.worker.archive_plan import archive_active_lock

    spec = load_runspec(_write_runspec(tmp_path))
    _write_archive_list(spec.paths.output_dir, ["a.tar.lz4"])
    _write_shard_config(spec.paths.output_dir, total_archives=1, shards_per_archive=2)
    archive = _make_archive(spec.paths.staging_dir, "a.tar.lz4")
    _make_clean_shard(spec.paths.output_dir, 0, model_count=1, s3_prefix=spec.storage.s3_output_prefix)
    _make_clean_shard(spec.paths.output_dir, 1, model_count=1, s3_prefix=spec.storage.s3_output_prefix)

    held = threading.Event()
    release = threading.Event()
    outcome: dict[str, object] = {}

    def worker() -> None:
        with archive_active_lock(spec.paths.output_dir, 0):
            held.set()
            release.wait(timeout=10)

    def cleanup() -> None:
        outcome["report"] = plan_archive_cleanup(spec, execute=True)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    assert held.wait(timeout=10)

    cleanup_thread = threading.Thread(target=cleanup, daemon=True)
    cleanup_thread.start()
    time.sleep(0.5)
    assert cleanup_thread.is_alive(), "cleanup should block on the worker-active lock"

    release.set()
    worker_thread.join(timeout=10)
    cleanup_thread.join(timeout=10)
    assert not cleanup_thread.is_alive()
    report = outcome["report"]
    assert isinstance(report, ArchiveCleanupReport)
    assert [item.status for item in report.items] == ["deleted"]
    assert not archive.exists()


def _write_runspec(tmp_path: Path, *, self_upload: bool = True, archive_source: str = "tracking") -> Path:
    data = {
        "dataset": {
            "name": "ds1",
            "run_id": "run",
            "mode": "archive",
            "array": "0-0",
            "archive_source": archive_source,
        },
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
            "self_upload": self_upload,
            "local_scratch": True,
            "scratch_dir": "/dev/shm",
            "s5cmd_path": "/bin/s5cmd",
            "upload_slots": 4,
        },
        "storage": {
            "s3_archive_prefix": "s3://example-bucket/structures/",
            "s3_output_prefix": "s3://example-bucket/users/test/postprocessed/ds1/",
            "gcs_destination_prefix": None,
            "allow_production_prefixes": False,
        },
        "validation": {
            "expected_archives": 1,
            "expected_one_archive_models": 2,
            "expected_one_archive_objects": 2,
            "expected_one_archive_aggregate_rows": 2,
        },
        "secrets": {"s3_credentials_ref": "env:bspp/s3", "gcs_credentials_ref": None},
    }
    path = tmp_path / "runspec.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _write_archive_list(output_dir: Path, names: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "archive_list.json").write_text(json.dumps(names))


def _write_shard_config(output_dir: Path, *, total_archives: int, shards_per_archive: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "archive_mode": True,
        "total_archives": total_archives,
        "shards_per_archive": shards_per_archive,
        "required_shards": total_archives,
        "array_range": f"0-{total_archives - 1}",
    }
    (output_dir / "shard_config.json").write_text(json.dumps(config, indent=2) + "\n")


def _make_archive(staging_dir: Path, name: str) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    archive = staging_dir / name
    archive.write_text("ARCHIVE\n")
    return archive


def _make_clean_shard(
    output_dir: Path,
    shard_id: int,
    *,
    model_count: int,
    s3_prefix: str,
    status: str = "uploaded",
    total_batches: int = 1,
    done_markers: int = 1,
    metadata_files: tuple[str, ...] = ("metadata.json",),
) -> None:
    shard = output_dir / f"shard_{shard_id}"
    shard.mkdir(parents=True, exist_ok=True)
    marker = {
        "s3_prefix": s3_prefix,
        "upload_mode": "files",
        "tar_prefix": "",
        "status": status,
        "total_files": 1,
        "total_batches": total_batches,
        "failed_models": 0,
        "metadata_files": list(metadata_files),
        "timestamp": "2026-09-11T00:00:00+00:00",
        "shard_id": shard_id,
        "model_count": model_count,
    }
    (shard / ".uploaded").write_text(json.dumps(marker))
    for batch_index in range(done_markers):
        (shard / f".batch_{batch_index}_done").touch()


def _write_allowlist(output_dir: Path, archive_name: str, count: int) -> None:
    allowlist_dir = output_dir / "allowlists"
    allowlist_dir.mkdir(parents=True, exist_ok=True)
    (allowlist_dir / f"{archive_name}.txt").write_text("\n".join(f"model_{i}" for i in range(count)) + "\n")
